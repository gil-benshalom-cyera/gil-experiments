import os.path
import time
import boto3
import logging
import json
import pandas as pd
import datetime
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError
from utils.funcs import init_logger, get_tools
from utils.decorators import log_execution_time
from batch_executor_base import BaseExecutorBase

logger = logging.getLogger(__name__)


# detailed structure of the formant here - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html


class BedrockAIBatchExecutor(BaseExecutorBase):

    def __init__(self, input_file: str, input_uri: str, output_uri: str, model_name: str, output_file_path: str,
                 final_output_path: str = 'output.csv', tools: list = None, tool_choice: dict = None,
                 location: str = None):
        super().__init__(
            input_file=input_file, input_uri=input_uri, output_uri=output_uri, model_name=model_name,
            output_file_path=output_file_path, final_output_path=final_output_path, tools=tools, tool_choice=tool_choice
        )
        timestamp = str(int(time.time()))
        self.job_name = f"batch-job-{timestamp}"
        self.bedrock = boto3.client(service_name="bedrock", region_name=location)
        self.s3_client = boto3.client('s3', region_name=location)
        self.input_data_config = {"s3InputDataConfig": {"s3Uri": self.input_uri}}
        self.output_data_config = {"s3OutputDataConfig": {"s3Uri": self.output_uri}}
        self.role_arn = "arn:aws:iam::374136425572:role/bedrock-service-role"

    def find_file_in_directory(self, bucket_name, directory_prefix, file_name):
        """
        Finds the complete path of a file in a specific directory in an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        :param directory_prefix: The directory prefix to search within.
        :param file_name: Name of the file to find.
        :return: The complete path of the file if found, otherwise None.
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=directory_prefix):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith(f"/{file_name}") or obj['Key'] == file_name:
                        return obj['Key']
            logger.info(f"File '{file_name}' not found in directory '{directory_prefix}' in bucket '{bucket_name}'.")
            return None
        except Exception as e:
            logger.info(f"An error occurred: {e}")
            return None

    def upload_to_s3(self, bucket_name, source_file_path, destination_blob_name):
        """
                Uploads a file to an S3 bucket.

                :param bucket_name: Name of the S3 bucket.
                :param source_file_path: Path to the file to be uploaded.
                :param destination_blob_name: Name of the file in the S3 bucket.
                """
        try:
            self.s3_client.upload_file(source_file_path, bucket_name, destination_blob_name)
            logger.info(f"File {source_file_path} uploaded to {bucket_name}/{destination_blob_name}")
        except FileNotFoundError:
            logger.info(f"The file {source_file_path} was not found.")
        except NoCredentialsError:
            logger.info("AWS credentials not available.")
        except Exception as e:
            logger.info(f"An error occurred: {e}")

    def download_from_s3(self, bucket_name, source_blob_name, destination_file_path):
        """
        Downloads a file from an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        :param source_blob_name: Name of the file in the S3 bucket.
        :param destination_file_path: Path to save the downloaded file.
        """
        try:
            self.s3_client.download_file(bucket_name, source_blob_name, destination_file_path)
            logger.info(f"File {source_blob_name} from {bucket_name} downloaded to {destination_file_path}")
        except FileNotFoundError:
            logger.info(f"The destination path {destination_file_path} was not found.")
        except NoCredentialsError:
            logger.info("AWS credentials not available.")
        except Exception as e:
            logger.info(f"An error occurred: {e}")

    def build_prompt(self, row: pd.Series) -> dict:
        prompt = {
            "recordId": row['custom_id'],
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": f"""please classify the following text to one of the following classes: {row['options']}. \n please provide a brief explanation for your choice.""",
                "messages": [{
                    "role": "user",
                    "content": row['text']
                }],
                "temperature": 0.2
            }
        }
        if self.tools and self.tool_choice:
            tools = []
            for tool in self.tools:
                tools.append({
                    'name': tool['name'],
                    'description': tool['description'],
                    'input_schema': tool['parameters']
                })
            prompt['modelInput']['tools'] = tools
            prompt['modelInput']['tool_choice'] = {
                'type': 'any',
                'name': self.tool_choice['name']
            }
        return prompt

    def upload_file(self) -> dict:
        bucket, blob = self.split_path(self.input_uri)
        self.upload_to_s3(bucket, self.input_file, blob)
        return {}

    def submit_job(self, **kwargs) -> dict:
        response = self.bedrock.create_model_invocation_job(
            roleArn=self.role_arn,
            modelId=self.model_name,
            jobName=self.job_name,
            inputDataConfig=self.input_data_config,
            outputDataConfig=self.output_data_config
        )
        job_arn = response.get('jobArn')
        logger.info(f"Job submitted with ARN: {job_arn}")
        return {'job_arn': job_arn}

    def wait_for_completion(self, job_arn: str, **kwargs) -> dict:
        status = self.bedrock.get_model_invocation_job(jobIdentifier=job_arn)['status'].lower()
        while status not in ['completed', 'failed']:
            logger.info(f'job status: {status}')
            time.sleep(30)
            status = self.bedrock.get_model_invocation_job(jobIdentifier=job_arn)['status'].lower()
        logger.info(f'job status: {status}')
        if status == 'failed':
            raise Exception("Job failed")
        return {'job_arn': job_arn, 'status': status}

    def download_result(self, job_arn: str, **kwargs) -> None:
        dir_suffix = job_arn.split('/')[-1]
        bucket, blob = self.split_path(self.output_uri)
        output_location = os.path.join(blob, dir_suffix, f'{self.input_file}.out')
        # output_location = self.find_file_in_directory(
        #     bucket_name=bucket, directory_prefix=blob, file_name=f'{self.input_file}.out'
        # )
        self.download_from_s3(bucket, output_location, self.output_file_path)

    def load_model_output(self, data: list) -> pd.DataFrame:
        rows = []
        with open(self.output_file_path, 'r') as f:
            for line in f:
                input_text, res = '', ''
                res = json.loads(line)
                try:
                    user_input = res['modelInput']['messages'][0]['content']
                    answer = res['modelOutput']['content'][0]['input']
                    rows.append({'input': user_input, 'output': answer})
                except:
                    logger.info(f"Error in processing row")
        df = pd.DataFrame(rows)
        return df

    # @log_execution_time
    # def run(self):
    #     input_data = self.get_input_list_for_model()
    #     # self.save_input_file(data=input_data)
    #     # input_res = self.upload_file()
    #     # job_res = self.submit_job(**input_res)
    #     completion_res = self.wait_for_completion(
    #         job_arn='arn:aws:bedrock:us-east-1:374136425572:model-invocation-job/7i8kjkhquvt0')
    #     self.download_result(**completion_res)
    #     df = self.load_model_output(data=input_data)
    #     df.to_csv(self.final_output_path, index=False)


if __name__ == '__main__':
    init_logger()
    load_dotenv(".env")
    _tools, _tool_choice = get_tools()

    executor = BedrockAIBatchExecutor(
        input_file='bedrock_sample_input.jsonl',
        input_uri=f"s3://batch-prediction-data/bedrock_sample_input.jsonl",
        output_uri="s3://batch-prediction-data/output_folder/",
        model_name="anthropic.claude-3-haiku-20240307-v1:0",
        output_file_path="bedrock_predictions.jsonl",
        tools=_tools,
        tool_choice=_tool_choice,
        location="us-east-1",
        final_output_path='output_bedrock.csv'
    )
    executor.run()

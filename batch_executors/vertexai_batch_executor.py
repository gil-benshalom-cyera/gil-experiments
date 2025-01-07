from batch_executor_base import BaseExecutorBase
import logging
import time
import pandas as pd
import vertexai
import json
from vertexai.batch_prediction import BatchPredictionJob
from utils.funcs import init_logger, get_tools
from google.cloud import storage

logger = logging.getLogger(__name__)


class VertexAIBatchExecutor(BaseExecutorBase):

    def __init__(self, input_file: str, input_uri: str, output_uri: str, model_name: str, output_file_path: str,
                 final_output_path: str = 'output.csv', tools: list = None, tool_choice: dict = None,
                 project: str = None, location: str = None):
        super().__init__(
            input_file=input_file, input_uri=input_uri, output_uri=output_uri, model_name=model_name,
            output_file_path=output_file_path, final_output_path=final_output_path, tools=tools,tool_choice=tool_choice
        )
        self.project = project
        self.location = location

    def upload_to_gcp(self, bucket_name, source_file_path, destination_blob_name):
        """
        Uploads a file to Google Cloud Storage.

        Args:
            bucket_name (str): The name of the GCP bucket.
            source_file_path (str): The path of the file to upload.
            destination_blob_name (str): The desired blob name in the bucket.

        Returns:
            str: Public URL of the uploaded file.
        """
        try:
            # Initialize the GCP storage client
            storage_client = storage.Client(project=self.project)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            # Upload the file
            blob.upload_from_filename(source_file_path)

            logger.info(f"File {source_file_path} uploaded to {destination_blob_name}.")
            return blob.public_url
        except Exception:
            logger.exception(f"Error uploading file")
            raise

    def download_from_gcp(self, bucket_name, source_blob_name, destination_file_path):
        """
        Downloads a file from Google Cloud Storage.

        Args:
            bucket_name (str): The name of the GCP bucket.
            source_blob_name (str): The name of the blob to download.
            destination_file_path (str): The path to save the downloaded file.

        Returns:
            None
        """
        try:
            # Initialize the GCP storage client
            storage_client = storage.Client(project=self.project)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)

            # Download the file
            blob.download_to_filename(destination_file_path)

            logger.info(f"File {source_blob_name} downloaded to {destination_file_path}.")
        except Exception as e:
            logger.info(f"Error downloading file: {e}")
            raise

    def build_prompt(self, row: pd.Series) -> dict:
        # TODO this can be edited in each task
        prompt = {
            "request": {
                "system_instruction": {
                    "parts": [
                        {
                            "text": "You *must* classify the following text into one of these classes:" + str(row['options']) + ". Even if uncertain, choose the most likely class and provide a brief explanation"
                        }
                    ]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{
                            "text": row['text']
                        }]
                    }
                ],
                'generation_config': {
                    'temperature': 0.2
                }
            }
        }
        if self.tools and self.tool_choice:
            prompt["request"]["tools"] = {"function_declarations": self.tools}
            prompt["request"]["tool_config"] = {
                "function_calling_config": {
                    "mode": "any", "allowed_function_names": [self.tool_choice["name"]]
                }
            }
        return prompt

    def upload_file(self) -> dict:
        bucket, blob = self.split_path(self.input_uri)
        self.upload_to_gcp(bucket, self.input_file, blob)
        return {}

    def submit_job(self, **kwargs) -> dict:
        vertexai.init(project=self.project, location=self.location)
        # Submit a batch prediction job with Gemini model
        batch_prediction_job = BatchPredictionJob.submit(
            source_model=self.model_name,
            input_dataset=self.input_uri,
            output_uri_prefix=self.output_uri,
        )

        # Check job status
        logger.info(f"Job resource name: {batch_prediction_job.resource_name}")
        logger.info(f"Model resource name with the job: {batch_prediction_job.model_name}")
        logger.info(f"Job state: {batch_prediction_job.state.name}")
        return {'batch_prediction_job': batch_prediction_job}

    def wait_for_completion(self, **kwargs) -> dict:
        batch_prediction_job = kwargs.get('batch_prediction_job')
        while not batch_prediction_job.has_ended:
            time.sleep(5)
            batch_prediction_job.refresh()

        # Check if the job succeeds
        if batch_prediction_job.has_succeeded:
            logger.info("Job succeeded!")
        else:
            logger.info(f"Job failed: {batch_prediction_job.error}")

        # Check the location of the output
        logger.info(f"Job output location: {batch_prediction_job.output_location}")
        return {'output_location': batch_prediction_job.output_location}

    def download_result(self, **kwargs) -> None:
        output_location = kwargs.get('output_location')
        bucket, blob = self.split_path(output_location)
        blob = f'{blob}/predictions.jsonl'
        self.download_from_gcp(bucket, blob, self.output_file_path)

    def load_model_output(self, data: list) -> pd.DataFrame:
        rows = []
        with open(self.output_file_path, 'r') as f:
            for line in f:
                input_text, res = '', ''
                res = json.loads(line)
                try:
                    input_text = res['request']['contents'][0]['parts'][0]['text']
                    answer = res['response']['candidates'][0]['content']['parts'][0]['functionCall']['args']['result']
                except:
                    logger.error(f'could not parse the response')
                    pass
                rows.append({'input_text': input_text, 'answer': answer})
        df = pd.DataFrame(rows)
        return df


if __name__ == '__main__':
    init_logger()
    _tools, _tool_choice = get_tools()

    executor = VertexAIBatchExecutor(
        input_file='sample_input.jsonl',
        input_uri=f"gs://gil_research/sample_input.jsonl",
        output_uri="gs://gil_research/output_folder/",
        model_name="gemini-1.5-flash-002",
        output_file_path="predictions.jsonl",
        tools=_tools,
        tool_choice=_tool_choice,
        project="prod-340608",
        location="us-central1",
        final_output_path='output_vertexai.csv'
    )
    executor.run()

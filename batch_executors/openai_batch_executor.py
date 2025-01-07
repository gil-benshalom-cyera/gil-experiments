from batch_executor_base import BaseExecutorBase
import os
from typing import Any
import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import BadRequestError
import logging
import time
import pandas as pd
import json
from utils.funcs import init_logger, get_tools

logger = logging.getLogger(__name__)


class OpenAIBatchExecutor(BaseExecutorBase):

    def __init__(self, input_file: str, model_name: str, output_file_path: str,
                 final_output_path: str = 'output.csv', tools: list = None, tool_choice: dict = None):

        super().__init__(
            input_file=input_file, model_name=model_name, output_file_path=output_file_path,
            final_output_path=final_output_path, tools=tools, tool_choice=tool_choice
        )
        self.client = self.init_client()

    def build_prompt(self, row: pd.Series) -> dict:
        prompt = {
            "custom_id": row['custom_id'],
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [
                    {"role": "system",
                     "content": f"""please classify the following text to one of the following classes: {row['options']}
                                        please provide a brief explanation for your choice."""},
                    {"role": "user", "content": row['text']}
                ]
            }
        }
        if self.tools and self.tool_choice:
            prompt['body']['functions'] = self.tools
            prompt['body']['function_call'] = self.tool_choice

        return prompt

    @staticmethod
    def init_client():
        client = AzureOpenAI(
            api_key=os.getenv("CYERA__LLM_CLIENT__AZURE_GPT40_MINI_BATCH_GLOBAL_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        return client

    def upload_file(self, **kwargs) -> dict:
        file = self.client.files.create(file=open(self.input_file, "rb"), purpose="batch")
        logger.info(file.model_dump_json(indent=2))
        file_id = file.id
        logger.info(file_id)
        return {'file_id': file_id}

    def submit_job(self, **kwargs) -> Any:
        max_tries = 3
        cnt = 0
        file_id = kwargs.get('file_id')
        while cnt < max_tries:
            try:
                batch_response = self.client.batches.create(
                    input_file_id=file_id,
                    endpoint="/chat/completions",
                    completion_window="24h",
                )

                # Save batch ID for later use
                batch_id = batch_response.id
                logger.info(batch_response.model_dump_json(indent=2))
                return {'batch_id': batch_id}
            except BadRequestError as e:
                if 'pending' in str(e):
                    cnt += 1
                    logger.info(f"File in status pending, trying again for {max_tries - cnt} more times")
                    time.sleep(20)

    def wait_for_completion(self, **kwargs) -> Any:
        batch_id = kwargs.get('batch_id')
        batch_response = self.client.batches.retrieve(batch_id)
        status = batch_response.status
        while status not in ("completed", "failed", "canceled"):
            logger.info(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")
            time.sleep(30)
            batch_response = self.client.batches.retrieve(batch_id)
            status = batch_response.status

        logger.info(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

        if batch_response and batch_response.status == "failed":
            for error in batch_response.errors.data:
                logger.info(f"Error code {error.code} Message {error.message}")

        return {'batch_response': batch_response}

    def log_errors(self, error_file_id: str):
        if error_file_id:
            file_response = self.client.files.content(error_file_id)
            raw_responses = file_response.text.strip().split('\n')
            for res in raw_responses:
                logger.warning(res)

    @staticmethod
    def process_row(row: dict):
        custom_id = row.get('custom_id')
        message = row.get('response').get('body').get('choices')[0].get('message')
        if function_call := message.get('function_call'):
            answer = function_call['arguments']
        else:
            answer = message.get('content')
        return {"custom_id": custom_id, "answer": answer}

    def download_result(self, **kwargs) -> None:
        batch_response = kwargs.get('batch_response')
        results = []
        output_file_id = batch_response.output_file_id
        error_file_id = batch_response.error_file_id
        self.log_errors(error_file_id=error_file_id)

        if not output_file_id:
            raise RuntimeError(f"Batch failed. Error file id: {output_file_id}")

        file_response = self.client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')

        for raw_response in raw_responses:
            if raw_response:
                results.append(self.process_row(row=json.loads(raw_response)))

        df = pd.DataFrame(results)
        df.to_csv(self.output_file_path, index=False)

    def load_model_output(self, data: list) -> pd.DataFrame:
        output_df = pd.read_csv(self.output_file_path)
        input_df = pd.DataFrame([{
            'custom_id': record['custom_id'],
            'system': record['body']['messages'][0]['content'],
            'user': record['body']['messages'][1]['content']
        } for record in data])
        if len(output_df) == 0:
            raise RuntimeError(f"Number of output rows is {len(output_df)}!")
        else:
            output = pd.merge(input_df, output_df, on='custom_id', how='left')
        return output


if __name__ == '__main__':
    init_logger()
    load_dotenv(".env")
    _tools, _tool_choice = get_tools()

    executor = OpenAIBatchExecutor(
        input_file='sample_input_openai.jsonl',
        model_name='gpt-4o-mini-batch',
        output_file_path="predictions_openai.csv",
        tools=_tools,
        tool_choice=_tool_choice,
        final_output_path='output_openai.csv'
    )
    executor.run()

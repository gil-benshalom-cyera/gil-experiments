import json
import pandas as pd
from abc import abstractmethod
from utils.decorators import log_execution_time


class BaseExecutorBase:

    def __init__(self, input_file: str, model_name: str, output_file_path: str, input_uri: str = None,
                 output_uri: str = None, final_output_path: str = 'output.csv', tools: list = None,
                 tool_choice: dict = None):
        self.input_file = input_file
        self.input_uri = input_uri
        self.output_uri = output_uri
        self.model_name = model_name
        self.output_file_path = output_file_path
        self.final_output_path = final_output_path
        self.tools = tools
        self.tool_choice = tool_choice

    @staticmethod
    def split_path(path: str):
        all_path = path.split('//')[1]
        parts = all_path.split('/')
        bucket = parts[0]
        blob = '/'.join(parts[1:])
        return bucket, blob

    def get_input_list_for_model(self) -> list:
        df = self.get_raw_data()
        df['custom_id'] = df.index.to_series().apply(lambda x: f'task-{x}')
        df['prompt'] = df.apply(lambda x: self.build_prompt(x), axis=1)
        return df['prompt'].tolist()

    @staticmethod
    def get_raw_data() -> pd.DataFrame:
        import requests
        input_rows = requests.get(
            'https://huggingface.co/datasets/AdaptLLM/law_knowledge_prob/resolve/main/test.jsonl').text.split('\n')
        del input_rows[-1]
        dict_rows = [json.loads(row) for row in input_rows]
        df = pd.DataFrame(dict_rows)
        # df = pd.concat([df] * 10, ignore_index=True)
        return df[0:100]

    def save_input_file(self, data: list):
        with open(self.input_file, "w") as file:
            for record in data:
                json_line = json.dumps(record)
                file.write(json_line + "\n")

    @abstractmethod
    def build_prompt(self, row: pd.Series) -> dict:
        pass

    @abstractmethod
    def upload_file(self) -> dict:
        pass

    @abstractmethod
    def submit_job(self, **kwargs) -> dict:
        pass

    @abstractmethod
    def wait_for_completion(self, **kwargs) -> dict:
        pass

    @abstractmethod
    def download_result(self, **kwargs) -> None:
        pass

    @abstractmethod
    def load_model_output(self, data: list) -> pd.DataFrame:
        pass

    @log_execution_time
    def run(self):
        input_data = self.get_input_list_for_model()
        self.save_input_file(data=input_data)
        input_res = self.upload_file()
        job_res = self.submit_job(**input_res)
        completion_res = self.wait_for_completion(**job_res)
        self.download_result(**completion_res)
        df = self.load_model_output(data=input_data)
        df.to_csv(self.final_output_path, index=False)

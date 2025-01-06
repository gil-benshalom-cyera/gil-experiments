import os
import json
import time
import datetime
import logging
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import BadRequestError
from pandas import Series
from utils.decorators import log_execution_time
from utils.funcs import init_logger, get_data, get_tools

load_dotenv(".env")
MODEL = 'gpt-4o-mini-batch'
INPUT_FILE_PATH = 'test.jsonl'
logger = logging.getLogger(__name__)


def get_dummy_data():
    return [
        {"custom_id": "task-0", "method": "POST", "url": "/chat/completions",
         "body": {"model": MODEL, "messages": [
             {"role": "system", "content": "You are an AI assistant that helps people find information."},
             {"role": "user", "content": "When was Microsoft founded?"}]}},
        {"custom_id": "task-1", "method": "POST", "url": "/chat/completions",
         "body": {"model": MODEL, "messages": [
             {"role": "system", "content": "You are an AI assistant that helps people find information."},
             {"role": "user", "content": "When was the first XBOX released?"}]}},
        {"custom_id": "task-2", "method": "POST", "url": "/chat/completions",
         "body": {"model": MODEL, "messages": [
             {"role": "system", "content": "You are an AI assistant that helps people find information."},
             {"role": "user", "content": "What is Altair Basic?"}]}}
    ]


def process_row(row: dict):
    custom_id = row.get('custom_id')
    message = row.get('response').get('body').get('choices')[0].get('message')
    if function_call := message.get('function_call'):
        answer = function_call['arguments']
    else:
        answer = message.get('content')
    return {"custom_id": custom_id, "answer": answer}


def init_client():
    client = AzureOpenAI(
        api_key=os.getenv("CYERA__LLM_CLIENT__AZURE_GPT40_MINI_BATCH_GLOBAL_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return client


def build_prompt(row: Series, tools: list, tool_choice: dict):
    prompt = {
        "custom_id": row['custom_id'],
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system",
                 "content": f"""please classify the following text to one of the following classes: {row['options']}
                                please provide a brief explanation for your choice."""},
                {"role": "user", "content": row['text']}
            ]
        }
    }
    if tools and tool_choice:
        prompt['body']['functions'] = tools
        prompt['body']['function_call'] = tool_choice

    return prompt


def get_input():
    df = get_data()
    df['custom_id'] = df.index.to_series().apply(lambda x: f'task-{x}')
    tools, tool_choice = get_tools()
    df['prompt'] = df.apply(lambda x: build_prompt(x, tools=tools, tool_choice=tool_choice), axis=1)
    return df['prompt'].tolist()[0:100]


def save_input_file(file_path: str):
    data = get_input()
    rows = []
    with open(file_path, "w") as file:
        for record in data:
            json_line = json.dumps(record)
            file.write(json_line + "\n")
            rows.append({
                'custom_id': record['custom_id'],
                'system': record['body']['messages'][0]['content'],
                'user': record['body']['messages'][1]['content']
            })
    df = pd.DataFrame(rows)
    return df


def upload_batch(file_path: str, client):
    # Upload a file with a purpose of "batch"
    file = client.files.create(file=open(file_path, "rb"), purpose="batch")

    logger.info(file.model_dump_json(indent=2))
    file_id = file.id
    logger.info(file_id)
    return file_id


def create_batch_job(file_id: str, client):
    max_tries = 3
    cnt = 0
    while cnt < max_tries:
        try:
            batch_response = client.batches.create(
                input_file_id=file_id,
                endpoint="/chat/completions",
                completion_window="24h",
            )

            # Save batch ID for later use
            batch_id = batch_response.id
            logger.info(batch_response.model_dump_json(indent=2))
            return batch_id
        except BadRequestError as e:
            if 'pending' in str(e):
                cnt += 1
                logger.info(f"File in status pending, trying again for {max_tries - cnt} more times")
                time.sleep(20)


def track_status(batch_id: str, client):
    batch_response = client.batches.retrieve(batch_id)
    status = batch_response.status
    while status not in ("completed", "failed", "canceled"):
        logger.info(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")
        time.sleep(30)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status

    logger.info(f"{datetime.datetime.now()} Batch Id: {batch_id},  Status: {status}")

    if batch_response and batch_response.status == "failed":
        for error in batch_response.errors.data:
            logger.info(f"Error code {error.code} Message {error.message}")

    return batch_response


def log_errors(error_file_id: str, client):
    if error_file_id:
        file_response = client.files.content(error_file_id)
        raw_responses = file_response.text.strip().split('\n')
        for res in raw_responses:
            logger.warning(res)


def retrieve_results(batch_response, client):
    results = []
    output_file_id = batch_response.output_file_id
    error_file_id = batch_response.error_file_id
    log_errors(error_file_id=error_file_id, client=client)

    if not output_file_id:
        raise RuntimeError(f"Batch failed. Error file id: {output_file_id}")

    file_response = client.files.content(output_file_id)
    raw_responses = file_response.text.strip().split('\n')

    for raw_response in raw_responses:
        if raw_response:
            results.append(process_row(row=json.loads(raw_response)))

    df = pd.DataFrame(results)
    return df


def save_to_file(input_df: pd.DataFrame, output_df: pd, batch_id: str):
    output = pd.merge(input_df, output_df, on='custom_id', how='left')
    output.to_csv(f'{batch_id}.csv', index=False)


def process_results(input_df: pd.DataFrame, batch_id: str, client):
    batch_response = track_status(batch_id=batch_id, client=client)
    output_df = retrieve_results(batch_response=batch_response, client=client)
    if len(output_df) == 0:
        raise RuntimeError(f"Number of output rows is {len(output_df)}!")
    else:
        save_to_file(input_df=input_df, output_df=output_df, batch_id=batch_id)


@log_execution_time
def process_batch_response(batch_id: str):
    client = init_client()
    input_df = save_input_file(file_path=INPUT_FILE_PATH)
    process_results(input_df=input_df, batch_id=batch_id, client=client)


@log_execution_time
def main():
    client = init_client()
    input_df = save_input_file(file_path=INPUT_FILE_PATH)
    file_id = upload_batch(file_path=INPUT_FILE_PATH, client=client)
    batch_id = create_batch_job(file_id=file_id, client=client)
    process_results(input_df=input_df, batch_id=batch_id, client=client)


if __name__ == '__main__':
    init_logger()
    # process_batch_response(batch_id='batch_5108ab14-c00b-48ff-8a01-ace7d6176ece')
    main()

import logging
import time

import pandas as pd
import vertexai
import json
from vertexai.batch_prediction import BatchPredictionJob
from pandas import Series
from utils.funcs import init_logger, get_data, get_tools
from utils.decorators import log_execution_time
from google.cloud import storage

PROJECT_ID = "prod-340608"
LOCATION = "us-central1"
INPUT_FILE = 'sample_input.jsonl'
INPUT_URI = f"gs://gil_research/{INPUT_FILE}"
OUTPUT_URI = "gs://gil_research/output_folder/"
MODEL_NAME = "gemini-1.5-flash-002"
OUTPUT_FILE_PATH = 'predictions.jsonl'

logger = logging.getLogger(__name__)


def upload_to_gcp(bucket_name, source_file_path, destination_blob_name):
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
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Upload the file
        blob.upload_from_filename(source_file_path)

        logger.info(f"File {source_file_path} uploaded to {destination_blob_name}.")
        return blob.public_url
    except Exception:
        logger.exception(f"Error uploading file")
        raise


def download_from_gcp(bucket_name, source_blob_name, destination_file_path):
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
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # Download the file
        blob.download_to_filename(destination_file_path)

        logger.info(f"File {source_blob_name} downloaded to {destination_file_path}.")
    except Exception as e:
        logger.info(f"Error downloading file: {e}")
        raise


def get_input():
    df = get_data()
    tools, tools_choice = get_tools()
    df['prompt'] = df.apply(lambda x: build_prompt(x, tools, tools_choice), axis=1)
    return df['prompt'].tolist()[0:100]


def build_prompt(row: Series, tools: list = None, tool_choice: dict = None):
    prompt = {
        "request": {
            "system_instruction": {
                "parts": [
                    {
                        "text": f"Please classify the following text to one of the following classes: {row['options']}.\n"
                                f"Choose one even if you are not sure.\n"
                                f"please provide a brief explanation for your choice."
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
            ]
        }
    }
    if tools and tool_choice:
        prompt["request"]["tools"] = {"function_declarations": tools}
        # prompt["tool_choice"] = tool_choice
    return prompt


def save_input_file(file_path: str):
    data = get_input()
    with open(file_path, "w") as file:
        for record in data:
            json_line = json.dumps(record)
            file.write(json_line + "\n")


def split_path(path: str):
    all_path = path.split('//')[1]
    parts = all_path.split('/')
    bucket = parts[0]
    blob = '/'.join(parts[1:])
    return bucket, blob


def upload_file(path: str):
    bucket, blob = split_path(path)
    upload_to_gcp(bucket, INPUT_FILE, blob)


def download_result(path: str, output_file_path: str):
    bucket, blob = split_path(path)
    blob = f'{blob}/predictions.jsonl'
    download_from_gcp(bucket, blob, output_file_path)


def wait_for_completion(batch_prediction_job):
    # Refresh the job until complete
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
    return batch_prediction_job.output_location


def submit_job(input_uri: str, output_uri: str):
    # Initialize vertexai
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Submit a batch prediction job with Gemini model
    batch_prediction_job = BatchPredictionJob.submit(
        source_model=MODEL_NAME,
        input_dataset=input_uri,
        output_uri_prefix=output_uri,
    )

    # Check job status
    logger.info(f"Job resource name: {batch_prediction_job.resource_name}")
    logger.info(f"Model resource name with the job: {batch_prediction_job.model_name}")
    logger.info(f"Job state: {batch_prediction_job.state.name}")
    return batch_prediction_job


def load_model_output(output_file_path: str):
    rows = []
    with open(output_file_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            input_text = res['request']['contents'][0]['parts'][0]['text']
            answer = res['response']['candidates'][0]['content']['parts'][0]['functionCall']['args']['result']
            rows.append({'input_text': input_text, 'answer': answer})
    df = pd.DataFrame(rows)
    return df


@log_execution_time
def main():
    save_input_file(file_path=INPUT_FILE)
    upload_file(path=INPUT_URI)
    batch_prediction_job = submit_job(input_uri=INPUT_URI, output_uri=OUTPUT_URI)
    remote_file_path = wait_for_completion(batch_prediction_job=batch_prediction_job)
    download_result(path=remote_file_path, output_file_path=OUTPUT_FILE_PATH)
    df = load_model_output(output_file_path=OUTPUT_FILE_PATH)
    df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    init_logger()
    # download_result('gs://gil_research/output_folder/prediction-model-2025-01-06T12:00:02.948994Z')
    main()
    # load_model_output(output_file_path=OUTPUT_FILE_PATH)

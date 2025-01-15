import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from datasets import Dataset
from utils import safe_literal_eval, get_timestamp
from tokenizer_config import *

tqdm.pandas()

CONTROL_CHARS_PATTERN = re.compile("[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u00a0]+")


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    timestamp = get_timestamp()
    detector_name = "flan_lastname"
    root_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(root_path, "output_models")
    model_name = detector_name + "_" + timestamp
    model_path = os.path.join(output_path, detector_name, model_name)

    # add model id and dataset path argument
    # TODO change to my dataset path
    dataset_path = '/Users/gilbenshalom/Github/cyera/libs/python/llm_client_py/train_results/1735029294/'
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model id to use for training. For example 'google/flan-t5-xl'")
    parser.add_argument("--dataset_path", type=str, default=dataset_path, help="Path to the dataset before processing.")
    parser.add_argument("--file_name", type=str, default="all_model_results.csv", help="Name of the dataset.")
    parser.add_argument("--output_dir", type=str, default=model_path, help="Path to the local fined tuned model.")
    parser.add_argument("--text_col", type=str, default="clean_text", help="Text column name (e.g., context column).")
    parser.add_argument("--last_name_tag_col", type=str, default="last_name", help="Last name column name.")
    args = parser.parse_known_args()
    return args


def preprocess_function(
        sample: Dataset,
        input_col: str,
        outputs_col: str,
        tokenizer: PreTrainedTokenizerBase,
        max_sample_length: int,
        max_target_length: int,
        padding="max_length"
) -> BatchEncoding:
    inputs = sample[input_col]
    model_inputs = tokenizer(text_target=inputs, max_length=max_sample_length, padding=padding, truncation=True)

    outputs = sample[outputs_col]
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=outputs, max_length=max_target_length + 20, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def config_output_str(row: pd.Series, last_name_tag_col: str) -> str:
    output_str = LABEL_PREFIXES["last_name"] + str(row[last_name_tag_col])
    # output_str = LABEL_PREFIXES["last_name"] + '[' + ','.join(row[last_name_tag_col]) + ']'
    return output_str


def clean_text(context):
    context = str(context)
    context = context.replace('\n', ' ')
    context = context.replace('\t', ' ')
    context = context.replace('\\r\\n', ' ')
    context = CONTROL_CHARS_PATTERN.sub(' ', context).strip()
    context = re.sub(' +', ' ', context)
    return context


def main():
    args, _ = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, model_max_length=512, clean_up_tokenization_spaces=True)
    max_sample_length = tokenizer.model_max_length
    print(f"Max input length: {max_sample_length}")

    df = pd.read_csv(os.path.join(args.dataset_path, args.file_name))
    df = df[[args.text_col, args.last_name_tag_col]]

    # eval cols
    col_lst = [args.text_col, args.last_name_tag_col]
    for col in col_lst:
        df[col] = df[col].progress_apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)

    inputs_col = "inputs"
    outputs_col = "outputs"

    # clean text
    # args.tokenized_text_col= args.text_col + "_clean"
    # df[args.tokenized_text_col] = df[args.text_col].progress_apply(lambda x: clean_text(x))
    # create prompted input
    df[inputs_col] = [TASK_PREFIX + str(doc) for doc in df[args.text_col]]
    # create output
    df[outputs_col] = df.progress_apply(
        lambda row: config_output_str(row, args.last_name_tag_col),
        axis=1
    )

    dataset = Dataset.from_pandas(df[[inputs_col, outputs_col]], preserve_index=False)

    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = dataset.map(
        lambda x: tokenizer(x[outputs_col], truncation=True), batched=True, remove_columns=[inputs_col, outputs_col]
    )
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    # use 95th percentile as max target length
    # max_target_length_95 = int(np.percentile(target_lenghts, 95))
    percentile_lst = [x / 10.0 for x in range(1, 10, 1)] + [0.95]
    percentiles = [int(np.percentile(target_lenghts, percentile)) for percentile in percentile_lst]
    max_target_length = max(percentiles) + 25  # 22+25
    print(f"Max target length: {max_target_length}")

    # process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=list(dataset.features),
        fn_kwargs={
            'input_col': inputs_col,
            'outputs_col': outputs_col,
            'tokenizer': tokenizer,
            'max_sample_length': max_sample_length,
            'max_target_length': max_target_length
        }
    )
    tokenized_dataset.save_to_disk(os.path.join(args.output_dir, "tokenized_dataset"))

    args.max_sample_length = max_sample_length
    args.max_target_length = max_target_length
    args.task_prefix = TASK_PREFIX
    args.label_prefixs = LABEL_PREFIXES
    args.label_separator = LABEL_SEPARATOR
    with open(os.path.join(args.output_dir, "tokenized_dataset_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    main()

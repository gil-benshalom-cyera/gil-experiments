import regex
from typing import Callable, Optional, Any, Literal
import time
from functools import wraps
from dataclasses import dataclass, field
import logging
import os

from transformers import PreTrainedTokenizer, AutoTokenizer
from ctranslate2 import Translator
from ctranslate2.converters import TransformersConverter

from flanner_trainer.ds_commons.consts import CONTROL_CHARS_PATTERN, DEFAULT_END_TOKEN


def clean_model_input(inputs: list[str]) -> list[str]:
    prepped_inputs_list = []
    for context in inputs:
        context = context.replace("\n", " ")
        context = context.replace("\t", " ")
        context = context.replace("\\r\\n", " ")
        context = CONTROL_CHARS_PATTERN.sub(" ", context).strip()
        context = regex.sub(" +", " ", context)
        prepped_inputs_list.append(context)
    return prepped_inputs_list


def ct2_inference(model_input: list[str],
                  tokenizer: PreTrainedTokenizer,
                  translator: Translator,
                  max_length: int,
                  generation_max_length: int,
                  batch_size: int,
                  end_token: str = DEFAULT_END_TOKEN,
                  num_hypotheses: int = 1,
                  sampling_temperature: float = 0,
                  sampling_topp: float = 1,
                  sampling_topk: int = 1,
                  beam_size: int = 1) -> list[str]:
    """
    Perform inference using CTranslate2 model.
    :param model_input: a list of input strings
    :param tokenizer: a tokenizer object (usually from Hugging Face Transformers)
    :param translator: a CTranslate2 translator object
    :param max_length: the maximum length of the input in tokens, the input will be truncated if it exceeds this length
    :param generation_max_length: the maximum length of the generated sequence in tokens
    :param batch_size: the size of the batch for inference in tokens
    :param end_token: a token that indicates the end of the sequence
    :param num_hypotheses:
    :param sampling_temperature:
    :param sampling_topp:
    :param sampling_topk:
    :param beam_size:
    :return:
    """
    token_ids_lst = tokenizer(model_input, truncation=True, max_length=max_length).input_ids
    tokens_lst = [tokenizer.convert_ids_to_tokens(token_ids) for token_ids in token_ids_lst]
    model_output = translator.translate_batch(tokens_lst, max_decoding_length=generation_max_length, num_hypotheses=num_hypotheses,
                                              max_input_length=max_length, max_batch_size=batch_size,
                                              sampling_temperature=sampling_temperature, sampling_topp=sampling_topp,
                                              sampling_topk=sampling_topk, beam_size=beam_size, end_token=end_token, batch_type='tokens')
    model_output = [tokenizer.convert_tokens_to_ids(sequence.hypotheses[0]) for sequence in model_output]
    model_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    return model_output


def convert_model_to_ct2(model_dir: str,
                         output_dir: str,
                         additional_files_to_copy: Optional[list[str]] = None,
                         quantization: Optional[Literal['int8_bfloat16', 'bfloat16']] = None) -> None:
    """
    Convert a model from Hugging Face Transformers to CTranslate2 format.
    :param model_dir: a directory with the model files
    :param output_dir: a directory where the converted model will be saved
    :param additional_files_to_copy: a list of additional files to copy except for the default ones which are:
                    ['special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json']
    :param quantization: a type of weight quantization, read more here: https://opennmt.net/CTranslate2/quantization.html
                            our flanners originally are 'bfloat16`.
    """
    files_to_copy = ['special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json']
    if additional_files_to_copy is not None:
        files_to_copy.extend(additional_files_to_copy)

    converter = TransformersConverter(model_name_or_path=model_dir, copy_files=files_to_copy)
    converter.convert(output_dir=output_dir, quantization=quantization)


def load_ct2_model(model_dir_path: str,
                   device: Literal['cpu', 'cuda'] = 'cuda',
                   device_index: int = 0,
                   compute_type: Literal['bfloat16', 'int8_bfloat16'] = 'bfloat16') -> Translator:
    """
    Load a CTranslate2 model.
    :param model_dir_path: a path to the model directory
    :param device: cpu or cuda
    :param device_index: the index of the GPU device
    :param compute_type: bfloat16 or int8_bfloat16
    :return: a CTranslate2 translator object (model)
    """
    return Translator(model_dir_path, device=device, device_index=device_index, compute_type=compute_type)


def add_task_prefix(model_input: list[str], task_prefix: str) -> list[str]:
    model_input_with_task = [f'{task_prefix}{x}' for x in model_input]
    return model_input_with_task


@dataclass
class LoggingKwargs:
    stage_name: Optional[str] = None
    rows_count: Optional[int] = None
    request_id: Optional[object] = None
    logger: Optional[logging.Logger] = None
    extra: dict[str, Any] = field(default_factory=dict)


def timing_decorator(predict_stage_func: Callable) -> Callable:
    @wraps(predict_stage_func)
    def wrapper(*args, logging_kwargs: LoggingKwargs = LoggingKwargs(), **kwargs):
        start_time = time.perf_counter()
        result = predict_stage_func(*args, **kwargs)
        end_time = time.perf_counter()

        if logging_kwargs.logger is None:
            return result

        execution_time = end_time - start_time
        execution_time_in_ms = 1000 * execution_time

        extra = {
            "execution_time": execution_time,
            "execution_time_in_ms": execution_time_in_ms,
            "stage_name": logging_kwargs.stage_name,
            **logging_kwargs.extra
        }
        msg = f'Finished {logging_kwargs.stage_name} in {execution_time} seconds'

        if logging_kwargs.rows_count is not None:
            extra["rows_count"] = logging_kwargs.rows_count
            extra["average_time_per_row_ms"] = execution_time_in_ms / logging_kwargs.rows_count
            msg += f' for {logging_kwargs.rows_count} rows'

        if logging_kwargs.request_id is not None:
            extra["request_id"] = logging_kwargs.request_id
            msg += f', request_id: {logging_kwargs.request_id}'

        logging_kwargs.logger.info(msg, extra=extra)

        return result
    return wrapper


def load_tokenizer(path: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.truncation_side = "right"
    return tokenizer


def download_s3_folder(bucket_name: str, s3_folder: str, local_dir: Optional[str] = None) -> None:
    # assumes credentials & configuration are handled outside python in .aws directory or environment variables
    import boto3
    session = boto3.Session(profile_name="user-provided-role")
    s3 = session.resource("s3")

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == "/":
            continue
        bucket.download_file(obj.key, target)


def set_bentoml_inference_log_level(log_level: int = logging.CRITICAL) -> None:
    """
    Controls log level for Bentoml prediction log.
    If log_level is ERROR, then request data will be logged together with the error.
    If log_level is INFO, then every request data and response data will be logged.
    :param log_level:
    """
    logging.getLogger('bentoml.prediction').setLevel(log_level)

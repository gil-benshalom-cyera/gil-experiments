import os
import yaml
import ast
import pandas as pd
from attr import dataclass
from datetime import datetime
from typing import Any, Dict, List, Union


class Config:
    def __init__(self, config_path: str, config_fn: str):
        with open(os.path.join(config_path, config_fn)) as stream:
            try:
                self.params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.ds_projects_path = os.path.join(os.path.expanduser("~"), "cyera", "ds-projects")
        self.project_path = os.path.join(self.ds_projects_path, self.params["project_name"])
        self.resources_path = os.path.join(self.project_path, self.params["resources_dir_name"])
        self.input_data_path = os.path.join(self.resources_path, self.params["input_data_dir_name"])
        self.detector_name = self.params["detector_name"]
        self.trial_number = self.params["trial_number"]
        self.description = self.params["description"]
        self.trial_path = os.path.join(self.input_data_path, self.detector_name, self.trial_number)
        self.annotated_path = os.path.join(self.trial_path, self.params["annotated_dir_name"])
        self.llm_annotated_path = os.path.join(self.trial_path, self.params["llm_annotated_dir_name"])
        self.data_fn = self.params["data_fn"]
        self.context_col_name = self.params["context_col_name"]
        self.full_name_col_name = self.params["full_name_col_name"]
        self.first_name_col_name = self.params["first_name_col_name"]
        self.last_name_col_name = self.params["last_name_col_name"]
        self.tool_calls_col_name = self.params["tool_calls_col_name"]
        self.result_tool_calls_key_name = self.params["result_tool_calls_key_name"]
        self.description_llm_temperature = self.params["description_llm_temperature"]
        self.annotation_id_col_name = self.params["annotation_id_col_name"]
        self.full_name_ground_truth_binary_col_name = self.full_name_col_name + "_ground_truth_binary"
        self.first_name_ground_truth_binary_col_name = self.first_name_col_name + "_ground_truth_binary"
        self.last_name_ground_truth_binary_col_name = self.last_name_col_name + "_ground_truth_binary"

        self.full_name_generated_binary_col_name = self.full_name_col_name + "_generated_binary"
        self.first_name_generated_binary_col_name = self.first_name_col_name + "_generated_binary"
        self.last_name_generated_binary_col_name = self.last_name_col_name + "_generated_binary"

        self.full_name_ground_truth_prep_col_name = self.full_name_col_name + "_ground_truth_prep"
        self.first_name_ground_truth_prep_col_name = self.first_name_col_name + "_ground_truth_prep"
        self.last_name_ground_truth_prep_col_name = self.last_name_col_name + "_ground_truth_prep"

        self.full_name_generated_prep_col_name = self.full_name_col_name + "_generated_prep"
        self.first_name_generated_prep_col_name = self.first_name_col_name + "_generated_prep"
        self.last_name_generated_prep_col_name = self.last_name_col_name + "_generated_prep"

        self.full_name_generated_parsed_col_name = self.full_name_col_name + "_generated_parsed"
        self.first_name_generated_parsed_col_name = self.first_name_col_name + "_generated_parsed"
        self.last_name_generated_parsed_col_name = self.last_name_col_name + "_generated_parsed"

        self.match_full_name_full_name_col_name = "match_full_name_full_name"
        self.match_first_name_first_name_col_name = "match_first_name_first_name"
        self.match_last_name_last_name_col_name = "match_last_name_last_name"
        self.match_full_name_first_name_col_name = "match_full_name_first_name"
        self.match_full_name_last_name_col_name = "match_full_name_last_name"
        self.match_first_name_full_name_col_name = "match_first_name_full_name"
        self.match_first_name_last_name_col_name = "match_first_name_last_name"
        self.match_last_name_full_name_col_name = "match_last_name_full_name"
        self.match_last_name_first_name_col_name = "match_last_name_first_name"


@dataclass
class GeneratedEntities:
    full_name_generated_parsed_lst: list
    first_name_generated_parsed_lst: list
    last_name_generated_parsed_lst: list
    full_name_generated_lst: list
    first_name_generated_lst: list
    last_name_generated_lst: list

@dataclass
class GroundTruthEntities:
    full_name_ground_truth_raw_lst: list
    first_name_ground_truth_raw_lst: list
    last_name_ground_truth_raw_lst: list
    full_name_ground_truth_lst: list
    first_name_ground_truth_lst: list
    last_name_ground_truth_lst: list

def eval_cols(col_lst: List[str], df: pd.DataFrame) -> pd.DataFrame:
    for col in col_lst:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    return df

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def parse_prompt_results(
        result: Dict[str, Any],
        full_name_col_name: str,
        first_name_col_name: str,
        last_name_col_name: str,
        full_name_generated_parsed: Union[List[str]]=[],
        first_name_generated_parsed: Union[List[str]]=[],
        last_name_generated_parsed: Union[List[str]]=[],
) -> (Union[str, List[str]], Union[str, List[str]], Union[str, List[str]]):
    assert isinstance(result, dict)
    try:
        result_key = list(result.keys())[0]
        if isinstance(result[result_key], dict):
            answer_keys = result[result_key].keys()
            if (full_name_col_name in answer_keys) and (
                    isinstance(result[result_key][full_name_col_name], (str, list))):
                full_name_generated_parsed = result[result_key][full_name_col_name]
            if (first_name_col_name in answer_keys) and (
                    isinstance(result[result_key][first_name_col_name], (str, list))):
                first_name_generated_parsed = result[result_key][first_name_col_name]
            if (last_name_col_name in answer_keys) and (
                    isinstance(result[result_key][last_name_col_name], (str, list))):
                last_name_generated_parsed = result[result_key][last_name_col_name]
    except IndexError:
        pass

    return full_name_generated_parsed, first_name_generated_parsed, last_name_generated_parsed


def get_timestamp():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    return timestamp

def replace_nan_in_empty_lst(df: pd.DataFrame, col_names_lst: List[str]):
    for col in col_names_lst:
        df[col] = df[col].apply(lambda x: [] if pd.isna(x) else x)

    return df

def eval_cols_safetly(df, col_names_lst):
    for col in col_names_lst:
        try:
            df[col] = df[col].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)
        except:
            pass
    return df

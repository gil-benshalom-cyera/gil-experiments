import json
import logging
import requests
import pandas as pd


def init_logger():
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Format for log messages
    )


def get_data():
    input_rows = requests.get(
        'https://huggingface.co/datasets/AdaptLLM/law_knowledge_prob/resolve/main/test.jsonl').text.split('\n')
    del input_rows[-1]
    dict_rows = [json.loads(row) for row in input_rows]
    df = pd.DataFrame(dict_rows)
    return df


def get_tools():
    tools = [
        {
            "name": "get_label",
            "description": "Returns label in a specific format",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "description": "A string that contains the label result"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "The explanation for the class that was chosen"
                            }
                        },
                        "required": ["label", "explanation"]
                    }
                }, "required": ["result"]
            }
        }
    ]

    tool_choice = {"name": "get_label"}
    return tools, tool_choice

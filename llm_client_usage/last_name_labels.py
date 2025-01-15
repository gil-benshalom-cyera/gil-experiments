import os
from datetime import datetime
import pandas as pd
from llm_client_py import get_one_model_predictions, DeploymentsGroupConfigs, LLMCredentials

DeploymentsGroupConfigs.GPT4O_MINI.tpm = 70000000
DeploymentsGroupConfigs.GPT4O_MINI.rpm = 54000

SAMPLE_SIZE = None
RESULT_BATCH_SIZE = 10000

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_person_name_entities",
            "description": "Gets the result of the entities of the NER task in the prompt. You are to return the output as valid JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "object",
                        "properties": {
                            "last_name": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of last names extracted from the NER task in the prompt."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "The explanation for the entities extracted from the NER task in the prompt."
                            },
                        }
                    }
                },
                "required": ["result"]
            },
        },
    }
]

tool_choice = {"type": "function", "function": {"name": "extract_person_name_entities"}}


def generate_sublists(input_list, size):
    """
    Generates sublists of a specified size from a given list.

    Args:
        input_list (list): The list to be split.
        size (int): The size of each sublist.

    Yields:
        list: Sublists of the specified size.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer.")
    for i in range(0, len(input_list), size):
        yield input_list[i:i + size]


def generate_df_subsets(df, size):
    """
    Generates subsets of a DataFrame with a specified number of rows.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        size (int): The number of rows in each subset.

    Yields:
        pd.DataFrame: Subsets of the original DataFrame.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer.")
    for i in range(0, len(df), size):
        sub_df = df.iloc[i:i + size].reset_index(drop=True)
        yield sub_df


def process_results(df_model_results: pd.DataFrame, input_df: pd.DataFrame, i: int, source_dir: str):
    df_model_results.to_csv(f'{source_dir}/raw_results_{i}.csv', index=False)
    df_model = df_model_results.copy()
    df_model['last_name'] = df_model['tool_calls'].apply(lambda x: x['result'].get('last_name', []))
    df_model['explanation'] = df_model['tool_calls'].apply(lambda x: x['result'].get('explanation', ''))
    df_model = df_model.merge(input_df[['clean_text']], left_index=True, right_index=True)
    df_model = df_model.dropna(subset=['tool_calls'])
    df_model[['clean_text', 'last_name', 'explanation']].to_csv(
        f'{source_dir}/model_results_{i}.csv', index=False
    )


def load_input_df(sample_size: int = None):
    # output_df = pd.DataFrame([
    #     {'clean_text': 'The consultation with Dr. Allon was very effective'},
    #     {'clean_text': 'My name is Amit Eshel'}
    # ])

    input_path = '/Users/gilbenshalom/Github/cyera/libs/python/auto_labeling_py/last_name_train_data.csv'
    df = pd.read_csv(input_path)
    output_df = df[['context']].rename(columns={'context': 'clean_text'})
    print(f'input size: {output_df.shape[0]}')
    output_df = output_df.drop_duplicates(subset=['clean_text'])
    print(f'input size after duplications: {output_df.shape[0]}')
    output_df = output_df.reset_index(drop=True)
    return output_df.head(sample_size) if sample_size else output_df


def create_messages(text_body):
    prompt = [
        {
            "role": "system",
            "content": """You are a Named Entity Recognition (NER) specialist tasked with accurately identifying **last names (last_name)** in text. Focus on high precision and recall, extracting only genuine last names related to individuals, while strictly avoiding company names, organization names, addresses, titles of works, or fictional characters. **Extract a name only if it clearly qualifies as a last name related to a person. If uncertain, return None.**### Key Definitions
- **last_name**: last names are always associated with a person, identified by the rest of the text, alongside titles such as 'Mr.'. 'Ms.', 'Miss', 'Misses', 'Dr.' etc., or in academic citation patterns. Only extract last names if the rest of the text confirms it refers to a person. If you are not sure you should return None. 
### Extraction Rules
#### Primary Indicators for Last Names
**Rule 1: Titles are Strong Indicators**
- **Titles for Last Names**: If a title (e.g., 'Mr.', 'Ms.', 'Misses', 'Miss', 'Dr.', 'Sergeant', etc.) appears directly before a name, always treat that name as a last name. You are not allowed to extract company or organizational names even if titles are present.
  - **Example**: For 'Dr. Carter', return `last_name: ['Carter']`.
**Rule 2: Citation Formats for Last Names**
- **Academic Citation Style for Multiple Last Names**: Recognize last names in academic citation formats, where last names appear with or without initials, such as 'Last, Initial.', 'Last Initial.', 'Last (Year)', or 'Last, First Initial'. Extract each unique last name from the sequence only if the last name is not part of a full name. 
  - **Example**: For 'Allon, A., Vixman, G., & Luria, R. (2018)', return `last_name: ['Allon', 'Vixman', 'Luria']`. For 'Allon & Luria (2017)' return 'last_name': ['Allon', 'Luria']. 
**Rule 3: Direct Mention of 'Last Name'**
- **Explicit Last Name Context**: If the phrase 'last name' is directly followed by a name, treat that name as a last name.
  - **Example**: For 'The last name is Johnson', return `last_name: ['Johnson']`.
#### Filtering for Precision in Extraction
**Rule 4: Exclude Company Names**
- Do not extract names if they appear in a context likely indicating a company (e.g., names followed by 'Inc.', 'LLP', 'Ltd.', etc.).
  - **Example**: For 'Client: Anderson Corp.', return `last_name: None`, For 'Client: 'Anderson and Allon' return 'last_name': ['Anderson', 'Allon'] .
**Rule 5: Avoid Extraction from Full Names and Unambiguous Patterns**
- **Full Name Patterns**: Avoid extracting last names if they appear as part of a full name pattern such as `First Last`, `Last, First`, or similar. These formats are full names rather than standalone last names.
  - **Example**: For 'Emily Davis', return `last_name: None`.
#### Context-Based Exclusions and Clarity
**Rule 6: Avoid Fictional, Famous Titles, and Book/Movie Names**
- **Exclude Known Titles and Characters**: Avoid extracting names that are clearly part of a title, famous character, or well-known work.
  - **Example**: For 'Sherlock Holmes in A Study in Scarlet', return `last_name: None`.
**Rule 7: Discern and Confirm Context for Personal Last Names**
- **Extract Only Confirmed Personal Last Names**: Extract a last name only if surrounding context strongly suggests it refers to a person and not an entity, job title, or address.
### Example-Based Guide
**Example 1**
Text: `Dr. Smith`
- **Explanation**: Let's take it step-by-steo. Step 1: Dr. is a title followed by a name. Step 2: titles followed by names are last name. Final step: 'Smith' is a last name related to an individual.
- **Extracted**: `last_name: ['Smith']`
**Example 2**
Text: `Client: Thompson Industries`
- **Explanation**: Let's take it step-by-step. Step 1: 'Industries' suggests 'Thompson Industries' is a company.
- Final step: **Extracted**: `last_name: None`
**Example 3**
Text: `The last name is Johnson`
- **Explanation**: Let's take it step-by-step. Step 1: The phrase 'last name is' indicates the text contains a last name. Step 2: The name 'Johnson' is a last name because it follows right after the phrase 'last name is'. Final step: **Extracted**: `last_name: ['Johnson']`
**Example 4**
Text: `Emily Davis`
- **Explanation**: Let's take it step-by-step. Step 1: 'Emily Davis' follows a full name structure `First Last`. Step 2: It is stricly not allowed to extract a last name if it is part of a full name. Final step: **Extracted**: `last_name: None`
**Example 5**
Text: `Allon, A., Vixman, G., & Luria, R.`
- **Explanation**: Let's take it step-by-step. Step 1: The text conatins a citation of an academic article. Step 2: Citation style indicates 'Allon', 'Vixman', and 'Luria' as last names in an academic context. Final step: **Extracted**: `last_name: ['Allon', 'Vixman', 'Luria']`
**Example 6**
Text: `Ms. Carter`
- **Explanation**: Let's take it step-by-step. Step 1: The title 'Ms.' before 'Carter' implies 'Carter' is a last name. Final step: **Extracted**: `last_name: ['Carter']`
**Example 7**
Text: `Harry Potter in The Sorcerer's Stone`
- **Explanation**: Let's take it step-by-step. Step 1: 'Harry Potter' is a fictional character, not a last name.
- **Extracted**: `last_name: None`
Ensure each extraction aligns with `last_name` or return None if context is ambiguous or unclear or if you are not sure.
Please verify yourself multiple times before giving your final answer. Good luck!
"""
        },
        {
            "role": "user",
            "content": """Based on this text, please extract all Last Name entities:
            {0}""".format(text_body)
        }
    ]
    return prompt


def get_list_to_process(df: pd.DataFrame) -> list:
    list_to_process = []
    for row in df.iterrows():
        row = row[1]
        list_to_process.append(create_messages(text_body=row['clean_text']))
    return list_to_process


def init_experiment():
    timestamp = str(int(round(datetime.now().timestamp())))
    input_df = load_input_df(sample_size=SAMPLE_SIZE)
    source_dir = f'train_results/{timestamp}'
    os.makedirs(source_dir, exist_ok=True)
    input_df.to_csv(f'{source_dir}/input_df.csv', index=False)
    return source_dir, input_df


def main():
    source_dir, input_df = init_experiment()
    for i, sub_df in enumerate(generate_df_subsets(df=input_df, size=RESULT_BATCH_SIZE)):
        list_to_process = get_list_to_process(sub_df)
        df_model_results = get_one_model_predictions(
            messages_list=list_to_process,  # Process only the current batch
            deployments_group=DeploymentsGroupConfigs.GEMINI_15FLASH_002,
            show_progress_bar=True,
            credentials=LLMCredentials(auto_auth_google_cloud=True),
            tool=tools[0],
            tool_choice=tool_choice,
            tempetature=0.25,
            # df_output_flavour=DFOutputFlavour.WIDE.value,
            batch_size=5000,
        )
        process_results(df_model_results, sub_df, i, source_dir=source_dir)


def concat_all_outputs(only_true_labels: bool = False):
    input_dir = '/Users/gilbenshalom/Github/cyera/libs/python/llm_client_py/train_results/1735029294'
    df = pd.DataFrame()
    for file in os.listdir(input_dir):
        if file.startswith('model_results'):
            tmp_df = pd.read_csv(os.path.join(input_dir, file))
            df = df.append(tmp_df)
    if only_true_labels:
        df['has_label'] = df['last_name'].apply(lambda x: 1 if x != '[]' else 0)
        df = df[df['has_label'] == 1]
        print(f'df size is {len(df)}...')
        df.to_csv(f'{input_dir}/all_model_results_with_labels.csv', index=False)
    else:
        print(f'df size is {len(df)}...')
        df.to_csv(f'{input_dir}/all_model_results.csv', index=False)
    return df


if __name__ == '__main__':
    # concat_all_outputs(only_true_labels=True)
    main()
import pandas as pd
from llm_client_py import get_one_model_predictions, DeploymentsGroupConfigs, LLMCredentials

list_to_process = [
    [
        {'role': 'user', 'content': f'Do you know Cyera?'},
        {'role': 'user', 'content': f'Can you elaborate more about the company?'}
    ]
]

cred = LLMCredentials(auto_auth_google_cloud=True)

results: pd.DataFrame = get_one_model_predictions(
    list_to_process,
    deployments_group=DeploymentsGroupConfigs.GPT4_TURBO,
    temperature=0.5,
    max_tokens=1000,
    credentials=cred
)

print(results)

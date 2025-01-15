from qwak_inference import BatchInferenceClient
import pandas as pd

batch_inference_client = BatchInferenceClient(model_id='t5_xlarge_last_name_gil_ct2')


def assign_additional_columns(df_input):
    for col in ['classification_name', 'classification_is_mock', 'classification_role', 'classification_tokenization',
                'classification_geo_locations', 'classification_role_origin']:
        df_input[col] = [None] * len(df_input)

    df_input.index.name = 'index_row'
    df_input = df_input.reset_index()
    return df_input

def load_sample_data():
    inputs = [
        'According to the paper written by Abramovich',
        'I went to Work with Gil Ben Shalom',
        'I got recommendations about Dr. Eshel',
        'I live at Bloch st'
    ]

    classifier_match = [
        'Abramovich',
        'Shalom',
        'Eshel',
        'Bloch'
    ]

    df_input = pd.DataFrame({'model_input': inputs, 'classifier_match': classifier_match})
    df_input = assign_additional_columns(df_input)
    return df_input


def load_validation_data():
    df = pd.read_csv('/Users/gilbenshalom/Github/cyera/ds-projects/flan_finetuned_py/val_prompt_dataset_final_corrected15.csv')
    df['model_input'] = df['context']
    df['classifier_match'] = df['text']
    df = df.drop(columns=[col for col in df.columns if col not in ['model_input', 'classifier_match']])
    df = assign_additional_columns(df)
    return df


def validate_results(df_input, df_results):
    df = df_input.merge(df_results, on='index_row')[
        ['index_row', 'context', 'full_name', 'last_name', 'model_generated_text', 'model_match', 'model_pred']
    ]
    df.to_csv('validation_results.csv', index=False)


def main():
    time_stamp = int(pd.Timestamp.now().timestamp())
    df_input = load_validation_data()

    df_results = batch_inference_client.run(
        df_input,
        batch_size=5000,
        executors=1,
        gpu_type='NVIDIA_A10G',
        gpus=1,
        serialization_format="parquet"
    )
    df_results.to_csv(f'results_{time_stamp}.csv', index=False)


def validate():
    df_input = pd.read_csv('val_prompt_dataset_final_corrected15.csv')
    df_input.index.name = 'index_row'
    df_input = df_input.reset_index()
    df_results = pd.read_csv('results_1735567777.csv')
    validate_results(df_input, df_results)


if __name__ == '__main__':
    main()
    # validate()

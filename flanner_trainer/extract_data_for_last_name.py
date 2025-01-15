from auto_labeling.config import cs
from auto_labeling.utils.db_utils.db_loader import fetch_as_dataframe_sf


query = '''
    SELECT * FROM flan_person_training_set_171124
'''

df = fetch_as_dataframe_sf(cs, query)

print(f'len of df: {len(df)}')

df.to_csv('last_name_train_data.csv', index=False)

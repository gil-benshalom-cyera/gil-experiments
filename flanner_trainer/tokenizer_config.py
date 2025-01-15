VERSION = 3
MODEL_ID = 'google/flan-t5-xl'

### hyperparameters
SEED = 42
EPOCHS = 3
BATCH_SIZE = 64
LABEL_PAD_TOKEN_ID = -100  # to ignore tokenizer pad token in the loss function calculation
GENERATION_MAX_LEN = 100
LEARNING_RATE = 3e-4  # 1e-4
GENERATION_NUM_BEAMS = 4

DEEP_SPEED_CONFIG_PATH = 'flan_t5_z3_config_bf16.json'

TOKENIZER_FOLDER_NAME = 'tokenizer'
ADD_TOKENS = ['\n', '\t', '\\r\\n']

### Output paths
DATASET_PATH = f'models/v{VERSION}/train_dataset'
MODEL_PATH = f'models/v{VERSION}/batch_size_{str(BATCH_SIZE)}'

### train from checkpoint
RESUME_CHECKPOINT_PATH = None
# RESUME_CHECKPOINT_PATH = f'{MODEL_PATH}/checkpoint-518'

TASK_PREFIXES = {
    1: "Classify the following document's sensitivity and category:",
    2: "document_sensitivity_category:",
    3: "ner_last_name:"
}

TASK_PREFIX = TASK_PREFIXES[VERSION]

LABEL_PREFIXES = {
    "full_name": "PER:",
    "first_name": "FN:",
    "last_name": "LN:",
}
LABEL_SEPARATOR = " #### "

TEXT_COL = "context"
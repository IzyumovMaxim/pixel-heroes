import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "mathbridge"
CONFIGS_DIR = ROOT_DIR / "configs"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TRAIN_DATASET_PATH = DATASET_DIR / "train.jsonl"
TEST_DATASET_PATH = DATASET_DIR / "test.jsonl"
RAW_TRAIN_DATASET_PATH = DATASET_DIR / "raw_train.jsonl"
PROCESSED_TRAIN_DATASET_PATH = DATASET_DIR / "processed_train.jsonl"
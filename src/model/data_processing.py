'''MathBridge dataset preparation'''
import os
import json
import random
import datasets

from config import load_config

from paths import DATASET_DIR, RAW_TRAIN_DATASET_PATH, PROCESSED_TRAIN_DATASET_PATH, TEST_DATASET_PATH, TRAIN_DATASET_PATH

config = load_config("data")

def load_data(num_examples: int = config["num_examples"]):
    '''Load the MathBridge dataset'''
    if os.path.exists(RAW_TRAIN_DATASET_PATH):
        return
    os.makedirs(DATASET_DIR)
    try:
        dataset = datasets.load_dataset(config["dataset"], split="train", streaming=True)
        subset = dataset.shuffle(seed=config["seed"]).take(config["num_examples"])
        with open(RAW_TRAIN_DATASET_PATH, "w") as f:
            for example in subset:
                    f.write(json.dumps(example) + "\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if os.path.exists(RAW_TRAIN_DATASET_PATH):
            os.rmdir(RAW_TRAIN_DATASET_PATH)

def process_math_bridge_data():
    '''Create training pairs from the MathBridge dataset'''
    if not os.path.exists(RAW_TRAIN_DATASET_PATH):
        print("MathBridge dataset not found")
        return
    try:
        with open(RAW_TRAIN_DATASET_PATH, "r") as source, open(PROCESSED_TRAIN_DATASET_PATH, "w") as processed:
            for line in source:
                raw_line = json.loads(line)
                equation = raw_line["equation"].replace("$", "").strip()
                spoken_equation = raw_line["spoken_English"]
                processed.write(json.dumps({"input": spoken_equation, "output": equation}) + "\n")
    except Exception as e:
        print(f"Error processing MathBridge dataset: {e}")

def train_test_split(test_size: float = config["test_size"]):
    '''Split the MathBridge dataset into training and testing sets'''
    if not os.path.exists(PROCESSED_TRAIN_DATASET_PATH):
        print("MathBridge dataset not found")
        return
    try:
        with open(PROCESSED_TRAIN_DATASET_PATH, "r") as source:
            lines = source.readlines()
            random.shuffle(lines)
            test_size = int(len(lines) * test_size)
            test_lines = lines[:test_size]
            train_lines = lines[test_size:]
            with open(TEST_DATASET_PATH, "w") as test, open(TRAIN_DATASET_PATH, "w") as train:
                for line in test_lines:
                    test.write(line)
                for line in train_lines:
                    train.write(line)
    except Exception as e:
        print(f"Error splitting MathBridge dataset: {e}")

if __name__ == "__main__":
    load_data()
    process_math_bridge_data()
    train_test_split()
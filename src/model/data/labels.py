'''MathBridge dataset preparation'''
import os
import json
import random
import datasets

MATH_BRIDGE_DATASET_PATH = "data/mathbridge"
MATH_BRIDGE_RAW_DATASET_PATH = os.path.join(MATH_BRIDGE_DATASET_PATH, "raw_train.jsonl")
MATH_BRIDGE_PROCESSED_DATASET_PATH = os.path.join(MATH_BRIDGE_DATASET_PATH, "processed_train.jsonl")
MATH_BRIDGE_TEST_DATASET_PATH = os.path.join(MATH_BRIDGE_DATASET_PATH, "test.jsonl")
MATH_BRIDGE_TRAIN_DATASET_PATH = os.path.join(MATH_BRIDGE_DATASET_PATH, "train.jsonl")

def load_data(num_examples: int = 10000):
    '''Load the MathBridge dataset'''
    if os.path.exists(MATH_BRIDGE_RAW_DATASET_PATH):
        return
    os.makedirs(MATH_BRIDGE_DATASET_PATH)
    try:
        dataset = datasets.load_dataset("aaai25withanonymous/MathBridge", split="train", streaming=True)
        subset = dataset.shuffle(seed=42).take(num_examples)
        with open(MATH_BRIDGE_RAW_DATASET_PATH, "w") as f:
            for example in subset:
                    f.write(json.dumps(example) + "\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if os.path.exists(MATH_BRIDGE_RAW_DATASET_PATH):
            os.rmdir(MATH_BRIDGE_RAW_DATASET_PATH)

def process_math_bridge_data():
    '''Create training pairs from the MathBridge dataset'''
    if not os.path.exists(MATH_BRIDGE_RAW_DATASET_PATH):
        print("MathBridge dataset not found")
        return
    try:
        with open(MATH_BRIDGE_RAW_DATASET_PATH, "r") as source, open(MATH_BRIDGE_PROCESSED_DATASET_PATH, "w") as processed:
            for line in source:
                raw_line = json.loads(line)
                equation = raw_line["equation"].replace("$", "").strip()
                spoken_equation = raw_line["spoken_English"]
                processed.write(json.dumps({"input": spoken_equation, "output": equation}) + "\n")
    except Exception as e:
        print(f"Error processing MathBridge dataset: {e}")

def train_test_split(test_size: float = 0.1):
    '''Split the MathBridge dataset into training and testing sets'''
    if not os.path.exists(MATH_BRIDGE_PROCESSED_DATASET_PATH):
        print("MathBridge dataset not found")
        return
    try:
        with open(MATH_BRIDGE_PROCESSED_DATASET_PATH, "r") as source:
            lines = source.readlines()
            random.shuffle(lines)
            test_size = int(len(lines) * test_size)
            test_lines = lines[:test_size]
            train_lines = lines[test_size:]
            with open(MATH_BRIDGE_TEST_DATASET_PATH, "w") as test, open(MATH_BRIDGE_TRAIN_DATASET_PATH, "w") as train:
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
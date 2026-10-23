'''Test model on test split of MathBridge dataset'''
import json
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)

def test_model(model, tokenizer, test_dataset_path: str, max_new_tokens: int = 100) -> float:
    '''Test model on test split of MathBridge dataset'''
    correct = 0
    incorrect = 0
    with open(test_dataset_path, "r") as f:
        progress =  tqdm.tqdm(f)
        for line in progress:
            example = json.loads(line)
            prompt = f"spoken: {example['input']}. latex: "
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
            text_output = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            if text_output.strip() == example['output'].strip():
                correct += 1
            else:
                incorrect += 1
            progress.set_postfix(correct=correct, incorrect=incorrect)
    return correct / (correct + incorrect)



if __name__ == "__main__":
    checkpoint = "HuggingFaceTB/SmolLM-135M"
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    exact_match = test_model(model, tokenizer, "data/mathbridge/test.jsonl")
    print(f"Exact match: {exact_match}")

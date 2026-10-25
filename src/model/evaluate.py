'''Test model on test split of MathBridge dataset'''
import json
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel

from config import load_config
from paths import TEST_DATASET_PATH, OUTPUTS_DIR

config = load_config("test")
config_train = load_config("train")
set_seed(config["seed"])

def test_model(model, tokenizer, max_new_tokens: int = config["max_new_tokens"]) -> float:
    '''Test model on test split of MathBridge dataset'''
    correct = 0
    incorrect = 0
    with open(TEST_DATASET_PATH, "r") as f:
        progress =  tqdm.tqdm(f)
        for line in progress:
            example = json.loads(line)
            prompt = f"spoken: {example['input']}. latex:"
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            text_output = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            if text_output.strip() == example['output'].strip():
                correct += 1
            else:
                incorrect += 1
            progress.set_postfix(correct=correct, incorrect=incorrect)
    return correct / (correct + incorrect)



if __name__ == "__main__":
    checkpoint = config["checkpoint"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(checkpoint)
    base_model.to(device)
    base_model.eval()

    base_exact_match = test_model(base_model, tokenizer)
    print(f"Base model exact match: {base_exact_match}")

    model = PeftModel.from_pretrained(base_model, OUTPUTS_DIR / config_train["adapter_dir"])
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    exact_match = test_model(model, tokenizer)
    print(f"Fine-tuned exact match: {exact_match}")

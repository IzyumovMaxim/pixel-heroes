from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed

from config import load_config, save_config
from paths import TRAIN_DATASET_PATH, TEST_DATASET_PATH, OUTPUTS_DIR

config = load_config("train")

checkpoint = config["checkpoint"]
set_seed(config["seed"])

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=config["lora"]["r"],
    lora_alpha=config["lora"]["lora_alpha"],
    target_modules=config["lora"]["target_modules"],
    lora_dropout=config["lora"]["lora_dropout"],
    bias=config["lora"]["bias"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = load_dataset("json", data_files=str(TRAIN_DATASET_PATH), split='train')
test_dataset = load_dataset("json", data_files=str(TEST_DATASET_PATH), split='train')

def tokenize_function(examples):
    prompt = f"spoken: {examples['input']}. latex:"
    answer = f' {examples["output"]}{tokenizer.eos_token}'
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    inputs_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": inputs_ids,
        "attention_mask": [1] * len(inputs_ids),
        "labels": labels,
    }

train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(tokenize_function, remove_columns=test_dataset.column_names)

collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

training_args = TrainingArguments(
    output_dir=OUTPUTS_DIR / config["adapter_dir"],
    eval_strategy="steps",
    learning_rate=config["training"]["learning_rate"],
    per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
    per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
    num_train_epochs=config["training"]["num_train_epochs"],
    weight_decay=config["training"]["weight_decay"],
    report_to=config["training"]["report_to"],
    eval_steps=config["training"]["eval_steps"],
    logging_steps=config["training"]["logging_steps"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
)

trainer.train()

model.save_pretrained(OUTPUTS_DIR / config["adapter_dir"])
save_config(config, "train", save_dir=OUTPUTS_DIR / config["adapter_dir"])
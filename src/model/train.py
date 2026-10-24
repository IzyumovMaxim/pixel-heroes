from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

checkpoint = "HuggingFaceTB/SmolLM-135M"

model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

train_dataset = load_dataset("json", data_files="data/mathbridge/train.jsonl", split='train')
test_dataset = load_dataset("json", data_files="data/mathbridge/test.jsonl", split='train')

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
    output_dir="outputs",
    eval_strategy="steps",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    eval_steps=50,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
)

trainer.train()

model.save_pretrained("outputs/lora_adapter")
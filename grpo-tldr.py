#!/usr/bin/env python3
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"
dataset = load_dataset("trl-lib/tldr", split="train")
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Target 10 words, 30-60 chars
def reward_len2_single(completion):
    words = len(completion.split())
    score1 = abs(10 - words)
    l = len(completion)
    if l > 60:
        score2 = 100 - (l-60)
    elif l < 30:
        score2 = 100 - 2 * (30 - l)
    else:
        score2 = 100
    return score1 + score2
        
def reward_len(completions, **kwargs):
    return [reward_len2_single(completion) for completion in completions]

def reformat(x):
    messages = [
        {"role": "user", "content": "Your role is to make a summary of the following post.\n" + x['prompt']}
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False)
    return {"prompt" : text}


dataset = dataset.map(
        reformat,
        remove_columns=dataset.column_names,
)

training_args = GRPOConfig(
    output_dir="Qwen3-0.6B-summary",
    logging_steps=10,
    save_strategy="steps",          # Save every X steps
    save_steps=50,                 # Save checkpoint every 500 steps
    save_total_limit=2,             # Keep only last 2 checkpoints
    report_to='wandb',
)
trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    peft_config = peft_config,
)
trainer.train()

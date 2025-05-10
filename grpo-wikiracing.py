#!/usr/bin/env python3
import re
from functools import lru_cache
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from collections import deque, defaultdict
import random

# Load dataset
def load_wiki_graph():
    dataset = load_dataset("HuggingFaceTB/simplewiki-pruned-350k")

    graph = defaultdict(list)
    titles = set()

    # Build the graph
    for example in dataset['train']:
        title = example['article']
        links = example['links']

        titles.add(title)

        for link in links:
            graph[title].append(link)

    return graph, titles

graph, titles = load_wiki_graph()
# BFS for shortest path
@lru_cache(maxsize=128*1024)
def shortest_path(source, dest):
    if source not in graph:
        print("Source not in graph")
    if dest not in graph:
        print("Dest not in graph")
    if source not in graph or dest not in graph:
        return None

    if source == dest:
        return [source]

    visited = {source}
    queue = deque([(source, 0)])

    while queue:
        current_node, l = queue.popleft()
        for neighbor in graph[current_node]:
            if neighbor == dest:
                return 1 + l
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, 1+l))

    return None  # No path found

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

def reward_single(completion, src, dst, pre_path_length, prompt):
    match = re.search(r"<answer>(.*)</answer>", completion)
    if not match:
        return -10
    try:
        node = int(match.group(1))
    except:
        # Not a parsable int
        return -8
    if node >= len(graph[src]):
        return -5
    chosen = graph[src][node]
    new_len = shortest_path(chosen, dst)
    # 1 if it moved in the right direction, 0 if it didn't change length, -1 if it made it worse
    reward = pre_path_length - new_len
    return reward
        
def reward_len(completions, **kwargs):
    return [reward_single(x, kwargs["src"][i], kwargs["dst"][i], kwargs['pre_path_length'][i], kwargs['prompts'][i]) for i,x in enumerate(completions)]

def reformat(x):
    current = x['src']
    target = x['dst']
    og_len = shortest_path(current, target)

    formatted_links = ""
    for i, x in enumerate(graph[current]):
        formatted_links += f"{i} {x}\n"
    formatted_path = ""

    m = f"""
You are playing WikiRun, trying to navigate from one Wikipedia article to another using only links.

IMPORTANT: You MUST put your final answer in <answer>NUMBER</answer> tags, where NUMBER is the link number.
For example, if you want to choose link 3, output <answer>3</answer>.

Current article: {current}
Target article: {target}
Available links (numbered):
{formatted_links}

Your path so far: {formatted_path}

Think about which link is most likely to lead you toward the target article.
First, analyze each link briefly and how it connects to your goal, then select the most promising one.

Remember to format your final answer by explicitly writing out the xml number tags like this: <answer>NUMBER</answer>
"""

    messages = [
        {"role": "user", "content": m}
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking = False)
    return {"prompt" : text, "pre_path_length": og_len}

random.seed(42)
titles = list(titles)
dataset = [{"src":random.choice(titles), "dst":random.choice(titles)} for x in range(50000)]
dataset = Dataset.from_list(dataset)

dataset = dataset.map(
        reformat,
        num_proc=16,
        ).filter(lambda x: x['pre_path_length'] is not None)


training_args = GRPOConfig(
    output_dir="Qwen3-0.6B-wikirun",
    logging_steps=10,
    save_strategy="steps",          # Save every X steps
    save_steps=50,                 # Save checkpoint every 500 steps
    save_total_limit=2,             # Keep only last 2 checkpoints
    num_train_epochs = 10,
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

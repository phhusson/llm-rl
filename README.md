# TLDR

Code: grpo-tldr.py

This LoRA aims to make the length of the TL;DR more reliable, targetting 10 words, and between 30 and 60 characters.

You can find the weight on [HuggingFace](https://huggingface.co/phh/Qwen3-0.6B-TLDR-Lora/)

This is a almost a copy/paste of HuggingFace GRPO example. 1100 steps are enough for a usable LoRA, which took 1hr40min on my RTX3090 for Qwen3-0.6B.

![reward curve of train](https://github.com/phhusson/llm-rl/raw/main/grpo-tldr.png "reward curve of train")

Usage:
```
<|im_start|>user

Your role is to make a summary of the following post.

insert <post here>

TL;DR:
<|im_end|>
<|im_start|>assistant
<think>

</think>

```


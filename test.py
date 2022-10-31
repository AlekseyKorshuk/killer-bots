import os
import torch
import numpy as np
from transformers import pipeline

import torch

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

pipe_flan = pipeline("text2text-generation", model="google/flan-t5-xl", device=0,
                     model_kwargs={"torch_dtype": torch.bfloat16})

with open('./killer_bots/bots/code_guru/database/handwritten.txt', 'r') as f:
    text = f.read()

paragraphs = text.split("\n")
paragraphs = [p for p in paragraphs if len(p) > 0]

for p in paragraphs:
    print(f"Paragraph: {p}")
    print(f"Response: {pipe_flan(f'Summarize the following text: {p}', max_length=4096)[0]['generated_text']}")
    print()

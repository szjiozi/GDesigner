import sys
sys.path.append("/workspace/juhao/adaptive_agent/GDesigner")
sys.stdout.reconfigure(encoding='utf-8')
from GDesigner.tools.reader.readers import JSONLReader
from datasets.gsm8k_dataset import gsm_data_process
import json
import random

dataset = JSONLReader.parse_file("datasets/gsm8k/gsm8k.jsonl")
dataset = gsm_data_process(dataset)
simple = []
medium = []
hard = []
simple_count = 0
medium_count = 0
hard_count = 0
for i in range(len(dataset)):
    step_num = len(dataset[i]["step"].split("\n"))
    if step_num <= 3:
        difficulty = "easy"
        dataset[i]["difficulty"] = difficulty
        simple.append(dataset[i])
    elif step_num <= 5:
        difficulty = "medium"
        dataset[i]["difficulty"] = difficulty
        medium.append(dataset[i])
    else:
        difficulty = "hard"
        dataset[i]["difficulty"] = difficulty
        hard.append(dataset[i])

processed_dataset = []
while True:
    processed_dataset.append(simple.pop(random.randint(0, len(simple)-1)))
    processed_dataset.append(medium.pop(random.randint(0, len(medium)-1)))
    processed_dataset.append(hard.pop(random.randint(0, len(hard)-1)))
    simple_count += 1
    medium_count += 1
    hard_count += 1
    if simple_count+medium_count+hard_count == 48:
        break
remaining_dataset = simple + medium + hard
random.shuffle(remaining_dataset)
processed_dataset.extend(remaining_dataset)
with open("datasets/gsm8k/gsm8k_processed.jsonl", "w",encoding="utf-8") as f:
    for item in processed_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
# simple = 0
# medium = 0
# hard = 0
# for i in range(40, 340):
#     if dataset[i]["difficulty"] == "easy":
#         simple += 1
#     elif dataset[i]["difficulty"] == "medium":
#         medium += 1
#     else:
#         hard += 1
# print(f"simple: {simple}, medium: {medium}, hard: {hard}")

# with open("datasets/gsm8k/gsm8k_difficulty.jsonl", "w",encoding="utf-8") as f:
#     for item in dataset:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")
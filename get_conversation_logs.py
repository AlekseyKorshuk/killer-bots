import json

with open("logs.txt", "r") as f:
    data = f.read()

data = data.split("\n")

conversations = []
for line in data:
    js = json.loads(line)
    if js["type"] == "whole_conversation":
        conversations.append(js)

for i, conversation in enumerate(conversations):
    print("#" * 14, i, "#" * 14)
    print(conversation["text"])
    print()

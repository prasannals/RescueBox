import pandas as pd
import json
import re

df = pd.read_csv("investigative_questions.csv", header=None)

questions = []
for row in df[0]:
    # Extract lines that look like numbered questions
    matches = re.findall(r'\d+\.\s*"?(.*?)"?$', row.strip())
    questions.extend(matches)

with open("preset_questions.json", "w") as f:
    json.dump({"questions": questions}, f, indent=2)

print(f"{len(questions)} questions saved to preset_questions.json")
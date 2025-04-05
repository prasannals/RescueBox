import json
import pandas as pd
import subprocess
import re
from sklearn.metrics import classification_report, confusion_matrix
import time

start_time = time.time()    


# === Load standardized conversation dataset ===
df = pd.read_csv("data/processed/combined_conversations.csv")

# === Load preset questions ===
with open("questions/preset_questions.json", "r") as f:
    questions = json.load(f)["questions"]

formatted_questions = "\n".join(f"- {q}" for q in questions)

# === Format prompt for Mistral ===
def format_prompt(conversation):
    return f"""
You are a message analyzer.

Based on the preset questions below:
{formatted_questions}

Analyze the following conversation:
{conversation}

Classify the conversation as:
- **True Positive (1):** The conversation contains strong indicators of criminal behavior based on the preset questions.
- **True Negative (0):** The conversation does not contain any signs of crime or suspicious behavior.
- **Ambiguous (2):** The conversation is unclear — there may be hints, but they are not strong enough to confidently classify as a crime.

If it is a True Positive, list the elements of crime you observe.

Respond in this format STRICTLY WITH CLASSIFICATION ONLY HAVING INTEGERS:
{{
  "classification": <0 | 1 | 2>,
  "elements_of_crime": ["<element1>", "<element2>"]
}}
"""

# === Send prompt to Mistral via Ollama ===
def call_mistral(prompt):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

def extract_json(response):
    try:
        # Use regex to find the first JSON-looking object
        match = re.search(r'\{[\s\S]*?\}', response)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print("Failed to parse JSON:", e)
    return None

predictions = []
elements = []

for _, row in df.iterrows():
    conv = row["conversation"]
    prompt = format_prompt(conv)
    response = call_mistral(prompt)
    parsed = extract_json(response)

    predictions.append(parsed.get("classification", None) if parsed else None)
    elements.append(parsed.get("elements_of_crime", []) if parsed else [])

df["prediction"] = predictions
df["elements_of_crime"] = elements
df.to_csv("predictions_with_elements.csv", index=False)

df = df.dropna(subset=["prediction"])

df["label"] = df["label"].astype(int)
df["prediction"] = df["prediction"].astype(int)

y_true = df["label"]
y_pred = df["prediction"]

print("Confusion Matrix: \n", confusion_matrix(y_true=y_true, y_pred=y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=3))

end_time = time.time()
total_time = end_time - start_time

print(f"⏱️ Total inference time: {total_time:.2f} seconds")
print(f"⏱️ Avg time per conversation: {total_time / len(df):.2f} seconds")
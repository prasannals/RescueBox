# Message-Analyzer

**Message Analyzer** is a forensic tool that analyzes text conversations to extract key criminal elements, with a special focus on murder-related analysis for now. The system uses **Ollama's Mistral 7B** model.


### Install dependencies

Run this in the root directory of the project:
```
poetry install
```

### Using the CLI
```
poetry run python src/message-analyser/message_analyser/main.py /message-analyzer/analyze "<input_file>,<output_directory>" "<elements of crime, separated by comma>"  
```



---

## Output Format
Your output will be a CSV with the following format:

| conversation_id | chunk_id | message_number | speaker | message                                      | crime_element |
|-----------------|----------|----------------|---------|----------------------------------------------|----------------|
| 1               | 2        | 48             | Marcus  | Just... some guys moving stuff. Quickly.     | Actus Reus     |

---
## Authors

1. Satya Srujana Pilli

2. Ashwini Ramesh Kumar 

3. Shalom Jaison

---





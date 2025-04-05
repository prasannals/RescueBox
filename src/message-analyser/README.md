# Message-Analyzer

**Message Analyzer** is a forensic tool that analyzes text conversations to extract key criminal elements, with a special focus on murder-related analysis for now. The system uses **Ollama's Mistral 7B** model and is deployed via a structured Flask-ML API.


The project includes:
- **Conversation-Based Crime Analysis**
- **Criminal Element Extraction**
- **Flask-ML API and Command Line Interface (CLI) support**
- **Ollama Inference: Uses Mistral 7B locally**
- **Chunked Inference: Handles long conversations by splitting into chunks of 30 messages**
- **Clean CSV Output with structured results**

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python **>=3.8**
- Pip **(latest version recommended)**
- Ollama installed locally and running
- Mistral 7B model pulled via:
```bash
ollama pull mistral:7b-instruct
```

### Clone the Repository
```bash
git clone https://github.com/mohanasrujana/Message-Analyzer.git
cd Message-Analyzer
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
📂 Message-Analyzer/
│
├── data/
|   ├── processed/
|   |   ├── combined_conversations.csv # Combined Dataset
|   ├── raw/
|   |   ├── true_negative_dataset/ #Prompt variables for true negative conversations
|   |   |   ├── augmented_true_negative_conversations.csv
|   |   |   ├── cities.txt
|   |   |   ├── conversation_topics.txt
|   |   |   ├── cross_validation_statistics.txt
|   |   |   ├── locations.txt
|   |   |   ├── mistral_validation_results.json
|   |   |   ├── participant_ages.txt
|   |   |   ├── participant_genders.txt
|   |   |   ├── participant_interests.txt
|   |   |   ├── participant_occupations.txt
|   |   |   ├── participant_personalities.txt
|   |   |   ├── time_settings.txt
|   |   |   ├── true_negative_conversations.csv
|   |   |   ├── true_negative_permutations.json
|   |   |   ├── true_negative_results.json
|   |   |   ├── true_negative_statistics.txt
|   |   ├── ambiguous_conversations.csv
|   |   ├── true_positives_conversations.csv
|   ├── combined_conversations_copy.csv
|   ├── predicted_result.csv
|   ├── error_log.txt
|
|
├── questions # (Not used in MVP; Plans to explore this option further down the line)
|   ├── investigative_questions.csv
|   ├── preset_questions.json
|
├── results/
|   ├── predicted_result.csv
|   ├── predicted_result_gemma_2b.csv
|   ├── raw_analysis.csv
|
├── scripts/
|   ├── api/
|   |   ├── cli.py
|   |   ├── server_info.md
|   |   ├── server.py
|   |   ├── server_gemma_2b.py
|   ├── jupyter_notebooks/
|   |   ├── Message_Analyser_Message_Generation.ipynb
|   |   ├── Message_Analyser_QuestionGeneration.ipynb
|   |   ├── Message_Analyzer_Ambiguous_Message_Generation.ipynb
|   |   ├── Message_Generation_True_Negative_.ipynb
|   |   ├── Mistral_Eval_True_Positives_.ipynb
|   |   ├── True_positives_ground_truth.ipynb
|   ├── export_to_onnx.py # (No longer required for MVP)
|   ├── extract_questions.py
|   ├── preprocessing.py
|   ├── run_mistral_inference.py
|   ├── run_onnx_inference.py
|
├── test/
|   ├── model_test.py # (No longer required for MVP)
|   ├── test_onnx_model.py # (No longer required for MVP)
|
├── .gitignore
├── requirements.txt  # Dependencies
├── README.md  # Documentation
```

---

## Key Components

### **Model Export Script (`export_to_onnx.py`)**
- Converts the modified **Gemma:2b** model to **ONNX format**.
- No longer used for MVP

### **ONNX Inference Script (`run_onnx_inference.py`)**
- Loads the ONNX model and performs inference on a given conversation.
- This was part of the product building process; No longer used/required for MVP

### **Command Line Interface (CLI) (`cli.py`)**
- Provides an easy-to-use CLI for message analyzisation.

### **Flask-ML API (`server.py`)**
- Deploys the model as a web API.

---


## Running Inference

### CLI Usage

CLI help

``` bash
python -m scripts.api.cli --help  
```

Predicting the conversations using cli:

Replace conversations_file with the input file of your conversations, results_dir with the directory in which you want your results, and "Actus Reus,Mens Rea" with the elements of crime you'd like to extract
```bash
python -m scripts.api.cli analyze --input_file [conversations_file] --output_file [results_dir] --elements_of_crime "Actus Reus,Mens Rea"
```

Here's the example command that worked for us:
```bash
python -m scripts.api.cli analyze --input_file data/combined_conversations_copy.csv --output_file results --elements_of_crime "Actus Reus,Mens Rea"
```


### API Usage
Start the Flask-ML API server:
```bash
python scripts/api/server.py
```
#### Server usage (method 1)
Once running, send a POST request manually on the terminal:
```bash
curl -X POST "http://127.0.0.1:5000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "input_file=@/path/to/conversations.csv" \
     -F "output_file=@/path/to/output/dir" \
     -F "elements_of_crime=Actus Reus,Mens Rea"
```

#### Server usage(method 2)
##### Use Rescue-Box-Desktop

- Install Rescue-Box from [link](https://github.com/UMass-Rescue/RescueBox-Desktop)
- Open Rescue-Box-Desktop and register the model by adding the server IP address and port number in which the server is running.
- Choose the model from list of available models under the **MODELS** tab.
- Checkout the Inspect page to learn more about using the model.
- Run the model. 
- View the output in Jobs
- Click on view to view the details and results



---

## Output Format
Your output will be a CSV with the following format:

| conversation_id | chunk_id | message_number | speaker | message                                      | crime_element |
|-----------------|----------|----------------|---------|----------------------------------------------|----------------|
| 1               | 2        | 48             | Marcus  | Just... some guys moving stuff. Quickly.     | Actus Reus     |

---

## Future Enhancements

1. **Add support for more models via toggle (e.g., Gemma ONNX, Mistral Ollama)** 

2. **Improve hallucination filtering and accuracy**

3. **Improve the gemma model to give accurate results**


## Authors

1. Satya Srujana Pilli

2. Ashwini Ramesh Kumar 

3. Shalom Jaison

---





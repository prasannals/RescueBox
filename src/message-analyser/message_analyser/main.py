import os
import time
import csv
import logging
import pandas as pd
import onnxruntime as ort
import re
from pathlib import Path
from typing import List, Dict, Any, TypedDict
from functools import lru_cache
from rb.lib.ml_service import MLService
from rb.api.models  import (
    ResponseBody, FileResponse, FileType, InputSchema, ParameterSchema, 
    InputType, TextParameterDescriptor, TaskSchema, FileInput, DirectoryInput
)
import ollama
import typer
from collections import defaultdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gemma-server')

APP_NAME = "message-analyzer"

# Initialize Flask-ML Server
server = MLService(APP_NAME)

# Create a singleton instance of the inference engine
@lru_cache(maxsize=1)
def get_inference_engine():
    """Get or create the inference engine singleton"""
    return GemmaOllamaInference()

# Define input and parameter types for Flask ML
class CrimeAnalysisInputs(TypedDict):
    """
    Defines the expected input structure for the crime analysis task.
    
    Attributes:
        input_file: Path to the CSV file containing conversations.
    """
    input_file: FileInput
    output_file: DirectoryInput

class CrimeAnalysisParameters(TypedDict):
    """
    Defines the parameters used for crime analysis.
    
    Attributes:
        include_all_messages: Whether to include all messages or only those with crime elements.
    """
    elements_of_crime: str

# Define the UI schema for the task
def create_crime_analysis_task_schema() -> TaskSchema:
    """
    Creates a schema for the criminal activity extraction task to define UI input fields and parameters.
    
    Returns:
        TaskSchema: A schema defining the required inputs and configurable parameters.
    """
    input_schema = InputSchema(
        key="input_file",
        label="CSV file containing conversations",
        input_type=InputType.FILE,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    
    elements_of_crime_schema = ParameterSchema(
        key="elements_of_crime",
        label="Elements of Crime",
        subtitle="Elements of Crime to be found in the Conversation",
        value=TextParameterDescriptor(
            default="Actus Reus,Mens Rea"
        )
    )
    
    return TaskSchema(
        inputs=[input_schema,output_schema],  # Only input is the CSV file
        parameters=[elements_of_crime_schema]
    )

# Add application metadata
server.add_app_metadata(
    name="Message Analyzer for Criminal Activity Extraction from Conversations",
    author="Satya Srujana Pilli, Shalom Jaison, Ashwini Ramesh Kumar",
    version="1.0.0",
    info="This application extracts and categorizes potential criminal activities from conversation text using a Gemma-2B ONNX model."
)

class GemmaOllamaInference:
    """
    Inference engine using Ollama with Gemma 2B for text generation
    """
    def __init__(self, model_name="mistral:7b-instruct"):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        try:
            print(prompt)
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}")
            return f"Error: {str(e)}"
    
    def extract_criminal_activity(self, conversation: str, crime_elements_str: str) -> str:
        """
        Extracts specified crime elements from a conversation using Ollama (Mistral).
        Ensures structured output only — no summaries, no interpretations.
        """

        crime_elements = [elem.strip() for elem in crime_elements_str.split(",") if elem.strip()]
        
        gold_examples = """
        Mens Rea:
        {Message Number: 31, Speaker: Marcus, content: "Definitely didn't intend any harm to 5. Corrections Officer."}
        {Message Number: 79, Speaker: Liam, content: "Maybe Marcus is just trying to shift the blame since he and 5. Corrections Officer argued about art."}

        Actus Reus:
        {Message Number: 48, Speaker: Marcus, content: "Just... some guys moving stuff. Quickly."}
        {Message Number: 70, Speaker: Marcus, content: "I'll handle the cleanup. Just… drop it."}
        """

        prompt = f"""
        You are a forensic conversation analyst. Your job is to extract ONLY the following crime elements from the conversation:

        CRIME ELEMENTS TO EXTRACT: {', '.join(crime_elements)}

        ---

        🧠 DO NOT:
        - Do NOT paraphrase, summarize, or explain.
        - Do NOT include bullets or markdown.
        - Do NOT include observations or conclusions.
        - Do NOT output if you're unsure — just leave that element blank.
        - If no message fits, return NOTHING.

        ---

        ✅ OUTPUT FORMAT:
        [List only the requested crime elements.]

        Mens Rea:
        {{Message Number: <num>, Speaker: <name>, content: "<exact message text>"}}

        Actus Reus:
        {{Message Number: <num>, Speaker: <name>, content: "<exact message text>"}}

        ❌ INCORRECT FORMATS:
        - Mens Rea: They sounded guilty.
        - Actus Reus: Someone did something illegal.

        ---

        📌 EXAMPLE:
        {gold_examples}

        ---

        Now analyze the following conversation:

        {conversation}

        ---

        EXTRACTED CRIMINAL ELEMENTS:
        """
        
        return self.generate_text(prompt)

    def parse_results_grouped(self, model_output: str, conversation_id: int, chunk_id: int) -> Dict[str, List[Dict]]:
        """
        Parses the model output and returns a dictionary grouping messages by crime element.
        """
        grouped_results = defaultdict(list)
        current_crime_element = None

        for line in model_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # Detect crime element section
            crime_element_match = re.match(r"^(Mens Rea|Actus Reus|Concurrence|Causation|Attempt|Complicity/Conspiracy|Obstruction of Justice|Extenuating Circumstances):$", line)
            if crime_element_match:
                current_crime_element = crime_element_match.group(1)
                continue

            # Match structured message line
            message_match = re.match(r"\{Message Number:\s*(\d+),\s*Speaker:\s*(.*?),\s*content:\s*\"(.*?)\"\}", line)
            if message_match and current_crime_element:
                message_number = int(message_match.group(1))
                speaker = message_match.group(2).strip()
                content = message_match.group(3).strip()

                grouped_results[current_crime_element].append({
                    "conversation_id": conversation_id,
                    "chunk_id": chunk_id,
                    "message_number": message_number,
                    "speaker": speaker,
                    "message": content
                })

        return dict(grouped_results)



def analyze_conversations(inputs: CrimeAnalysisInputs, parameters: CrimeAnalysisParameters) -> ResponseBody:
    """
    Process a CSV file containing conversations, extract criminal activities, and save results.
    """
    start_time = time.time()
    temp_file = None
    
    RESULTS_DIR = Path(inputs["output_file"].path)
    RESULTS_FILE = RESULTS_DIR / "predicted_result.csv"
    try:
        # Get parameters
        crime_elements = parameters.get("elements_of_crime", "Actus Reus,Mens Rea")
        
        # Debug the input object
        file_input = inputs["input_file"]
        logger.info(f"Input file object type: {type(file_input)}")
        logger.info(f"Input file object attributes: {dir(file_input)}")
        
        # Create a temporary file to work with
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_path = temp_file.name
        logger.info(f"Created temporary file: {temp_path}")
        
        # Try to extract the file content
        try:
            # access the file attribute
            if hasattr(file_input, 'file'):
                logger.info("Using 'file' attribute")
                # Read the file content and write to temp file
                content = file_input.file.read()
                temp_file.write(content)
                temp_file.flush()
            # access the path attribute
            elif hasattr(file_input, 'path'):
                logger.info(f"Using 'path' attribute: {file_input.path}")
                # Copy the file to temp location
                import shutil
                shutil.copyfile(file_input.path, temp_path)
            # Try direct read method
            elif hasattr(file_input, 'read'):
                logger.info("Using 'read' method")
                content = file_input.read()
                temp_file.write(content)
                temp_file.flush()
            # Try accessing the flask request files
            else:
                from flask import request
                if hasattr(request, 'files') and 'input_file' in request.files:
                    logger.info("Using Flask request.files")
                    file = request.files['input_file']
                    file.save(temp_path)
                else:
                    raise ValueError("Could not extract file content from FileInput object")
                    
            # Close the temp file to ensure all data is written
            temp_file.close()
            
            # Now read the CSV from the temporary file
            logger.info(f"Reading CSV from temporary file: {temp_path}")
            df = pd.read_csv(temp_path)
            
        except Exception as e:
            logger.error(f"Error extracting file content: {str(e)}")
            raise ValueError(f"Could not process file: {str(e)}")
            
        # Ensure the CSV contains a conversation column
        if "conversation" not in df.columns:
            raise ValueError("CSV file must contain a 'conversation' column")
        
        # Get the inference engine
        engine = get_inference_engine()
        
        # Process each conversation
        results = []
        raw_outputs = []


        
        for i, row in df.iterrows():
            conversation = row["conversation"]
            logger.info(f"Processing conversation {i+1}/{len(df)}")

            messages = conversation.strip().split("\n")
            chunks = [messages[k:k + 30] for k in range(0, len(messages), 30)]
            # Extract criminal activity
            for j, chunk in enumerate(chunks):
                chunk_text = "\n".join(chunk)
                raw_output = engine.extract_criminal_activity(chunk_text, crime_elements)
                logger.info(f"Raw output: {raw_output}")

                grouped_result = engine.parse_results_grouped(raw_output, conversation_id=i+1, chunk_id=j+1)
                for crime_element, messages in grouped_result.items():
                    for res in messages:
                        res["crime_element"] = crime_element
                        results.append(res)
                
            logger.info(f"Completed conversation {i+1}, found {len(results)} relevant messages")

        # Save results to the fixed output path
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save CSV with all extracted messages
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["conversation_id", "chunk_id", "message_number", "speaker", "message", "crime_element"])
            writer.writeheader()
            writer.writerows(results)
        
        # Return the file response
        logger.info(f"Analysis completed in {time.time() - start_time:.2f}s. Results saved to {RESULTS_FILE}")
        
        return ResponseBody(FileResponse(path=str(RESULTS_FILE), file_type=FileType.CSV))
        
    except Exception as e:
        logger.error(f"Error analyzing conversations: {str(e)}")
        
        # Create error log file
        error_file = Path("error_log.txt")
        if RESULTS_DIR:
            error_file = RESULTS_DIR / "error_log.txt"
        with open(error_file, "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Include stack trace
            import traceback
            f.write("\nStack trace:\n")
            f.write(traceback.format_exc())
        
        return ResponseBody(FileResponse(path=str(error_file), file_type=FileType.TEXT))
        
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"Removed temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

def cli_parser(arg: str) -> dict:
    """
    Parse a single string argument into the input dictionary.
    
    Expected format: "input_file_path,output_directory_path"
    """
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(
            "Expected a single argument with two comma-separated paths: "
            "input_file_path,output_directory_path"
        )
    input_file_path = parts[0].strip()
    output_directory_path = parts[1].strip()
    return {
        "input_file": FileInput(path=input_file_path),
        "output_file": DirectoryInput(path=output_directory_path)
    }

# Define the CLI parser for parameters (optional).
def param_parser(arg: str) -> dict:
    """
    Parse a single string argument into the parameters dictionary.
    
    Expected: A comma-separated string of crime elements.
    Defaults to "Actus Reus,Mens Rea" if empty.
    """
    crime_elements = arg.strip() if arg.strip() else "Actus Reus,Mens Rea"
    return {"elements_of_crime": crime_elements}

server.add_ml_service(
    rule="/analyze",
    ml_function=analyze_conversations,
    inputs_cli_parser=typer.Argument(
        parser=cli_parser,
        help="Comma-separated paths: <input_file_path>,<output_directory_path>"
    ),
    parameters_cli_parser=typer.Argument(
        parser=param_parser,
        help="Comma-separated crime elements (e.g., 'Actus Reus,Mens Rea')"
    ),
    short_title="Criminal Activity Extraction",
    order=0,
    task_schema_func=create_crime_analysis_task_schema,
)


app = server.app

# Run the server
if __name__ == "__main__":
    print("Starting Message Analysis Server...")
    # Start the server
    app()
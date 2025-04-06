import os
import time
import csv
import json
import logging
import pandas as pd
import numpy as np
import onnxruntime as ort
import re
from pathlib import Path
from typing_extensions import List, Dict, Any, TypedDict
from functools import lru_cache
from flask import jsonify, request
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    ResponseBody, FileResponse, FileType, InputSchema, ParameterSchema, 
    InputType, EnumParameterDescriptor, EnumVal, TaskSchema, FileInput, DirectoryInput
)
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gemma-server')

# Initialize Flask-ML Server
server = MLServer(__name__)

# Configuration
MODEL_PATH = "models/gemma-2b.onnx"
TOKENIZER_NAME = "google/gemma-2b"
MAX_LENGTH = 2048
NUM_THREADS = 4

class GemmaOnnxInference:
    """
    Optimized ONNX inference engine for Gemma-2B model with focus on text generation
    for criminal activity extraction from conversations.
    """
    
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        tokenizer_name: str = TOKENIZER_NAME,
        max_length: int = MAX_LENGTH,
        num_threads: int = NUM_THREADS
    ):
        """
        Initialize the ONNX inference engine.
        
        Args:
            model_path: Path to the ONNX model file
            tokenizer_name: Hugging Face tokenizer name or path
            max_length: Maximum sequence length for generation
            num_threads: Number of threads for inference
        """
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.num_threads = num_threads
        
        # Initialize components
        self._validate_model()
        self.tokenizer = self._load_tokenizer()
        self.session = self._create_session()
        
        logger.info(f"Gemma ONNX inference engine initialized with model: {model_path}")
    
    def _validate_model(self) -> None:
        """Validate that the ONNX model file exists."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found at: {self.model_path}")
        
        # Check for external data files
        model_dir = os.path.dirname(self.model_path)
        data_file = os.path.join(model_dir, "gemma-2b_data.bin")
        weights_file = os.path.join(model_dir, "gemma-2b-weights.bin")
        
        if not (os.path.exists(data_file) or os.path.exists(weights_file)):
            logger.warning(
                "External data files not found. Model may not load correctly. "
                "Ensure external data files are present in the model directory."
            )
    
    def _load_tokenizer(self):
        """Load the Hugging Face tokenizer for the model."""
        try:
            logger.info(f"Loading tokenizer: {self.tokenizer_name}")
            return AutoTokenizer.from_pretrained(self.tokenizer_name)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise RuntimeError(f"Tokenizer initialization failed: {str(e)}")
    
    def _create_session(self):
        """Create an optimized ONNX Runtime session."""
        try:
            # Configure session options for optimal performance
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = self.num_threads
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            # Enable memory optimizations
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            
            # Select appropriate execution providers
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
                logger.info("Using CUDA acceleration for inference")
            
            # Create and return the session
            logger.info(f"Creating ONNX Runtime session with providers: {providers}")
            return ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {str(e)}")
            raise RuntimeError(f"ONNX session initialization failed: {str(e)}")
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using the ONNX model for a single prompt.
        
        Args:
            prompt: The input prompt for text generation
            
        Returns:
            Generated text response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Convert to numpy arrays for ONNX Runtime
            onnx_inputs = {
                name: tensor.numpy() for name, tensor in inputs.items()
            }
            
            # Run inference
            start_time = time.time()
            outputs = self.session.run(None, onnx_inputs)
            inference_time = time.time() - start_time
            
            # Process outputs - get logits from the model output
            logits = outputs[0]
            
            # Get the most likely tokens (simple greedy decoding)
            predictions = np.argmax(logits, axis=-1)
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            
            logger.info(f"Text generated in {inference_time:.2f}s")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def extract_criminal_activity(self, conversation: str) -> str:
        """
        Format the prompt and extract criminal activity from a conversation.
        
        Args:
            conversation: The conversation text to analyze
            
        Returns:
            Generated analysis of criminal elements in the conversation
        """
        prompt = """You are a forensic conversation analyst tasked with extracting potential criminal activities from the provided conversation.

        INSTRUCTIONS:
        1. Analyze the conversation for messages that indicate possible criminal activity
        2. For each relevant message, extract the speaker, message content, and categorize under ONE appropriate crime element
        3. Format EACH extracted message EXACTLY as follows:
        - Message from [Speaker]: "[Exact message text]" | Crime element: [Category]
        4. ONLY include messages that contain criminal elements
        5. If a message doesn't clearly indicate criminal activity, DO NOT include it
        6. DO NOT modify or paraphrase the original messages
        7. DO NOT include any explanations or commentary outside the specified format
        8. DO NOT include HTML or markdown formatting in your response

        CRIME ELEMENT CATEGORIES (use only these exact categories):
        - Actus Reus (Criminal Act)
        - Mens Rea (Guilty Mind)
        - Concurrence
        - Causation
        - Attempt
        - Complicity/Conspiracy
        - Obstruction of Justice
        - Extenuating Circumstances

        EXAMPLE OUTPUT:
        - Message from John: "I knew it was illegal but I did it anyway" | Crime element: Mens Rea (Guilty Mind)
        - Message from Sarah: "We planned the break-in together last week" | Crime element: Complicity/Conspiracy

        CONVERSATION TO ANALYZE:
        {0}

        EXTRACTED CRIMINAL ELEMENTS:""".format(conversation)
        
        # Generate text with the formatted prompt
        return self.generate_text(prompt)
    
    def parse_results(self, raw_output: str, include_all: bool = False) -> List[Dict[str, str]]:
        """
        Parse the raw output from the model into structured data.
        
        Args:
            raw_output: Raw output from the model
            include_all: Whether to include messages with no crime elements
            
        Returns:
            list: List of dictionaries containing parsed messages and crime elements
        """
        # Define regex pattern for properly formatted responses
        pattern = r'- Message from (.+?): "(.+?)" \| Crime element: (.+?)$'
        
        # Extract all matches
        results = []
        for line in raw_output.strip().split('\n'):
            if line and not line.isspace():
                match = re.match(pattern, line)
                if match:
                    speaker = match.group(1).strip()
                    message = match.group(2).strip()
                    category = match.group(3).strip()
                    
                    # Only include relevant messages based on parameter
                    if include_all or category.lower() != "none":
                        results.append({
                            "speaker": speaker,
                            "message": message,
                            "crime_element": category
                        })
        
        return results

# Create a singleton instance of the inference engine
@lru_cache(maxsize=1)
def get_inference_engine():
    """Get or create the inference engine singleton"""
    return GemmaOnnxInference(
        model_path=MODEL_PATH,
        tokenizer_name=TOKENIZER_NAME,
        max_length=MAX_LENGTH,
        num_threads=NUM_THREADS
    )

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
    include_all_messages: str

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
    
    include_all_messages_schema = ParameterSchema(
        key="include_all_messages",
        label="Include All Messages",
        subtitle="Include messages without criminal elements in the output",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(key="false", label="False (Only Criminal Elements)"),
                EnumVal(key="true", label="True (All Messages)"),
            ],
            default="false"
        )
    )
    
    return TaskSchema(
        inputs=[input_schema,output_schema],  # Only input is the CSV file
        parameters=[include_all_messages_schema]
    )

@server.route("/analyze", task_schema_func=create_crime_analysis_task_schema, short_title=" Message Analysis", order=0)
def analyze_conversations(inputs: CrimeAnalysisInputs, parameters: CrimeAnalysisParameters) -> ResponseBody:
    """
    Process a CSV file containing conversations, extract criminal activities, and save results.
    """
    start_time = time.time()
    temp_file = None
    
    try:
        # Get parameters
        include_all = parameters.get("include_all_messages", "false") == "true"
        
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

        RESULTS_DIR = Path(inputs["output_file"].path)
        RESULTS_FILE = RESULTS_DIR / "predicted_result.csv"
        
        for i, row in df.iterrows():
            conversation = row["conversation"]
            logger.info(f"Processing conversation {i+1}/{len(df)}")
            #logger.info(f"conversation: {conversation}")
            
            # Extract criminal activity
            raw_output = engine.extract_criminal_activity(conversation)
            logger.info(f"Raw output: {raw_output}")
            raw_outputs.append(raw_output)
            
            # Parse the results
            parsed_results = engine.parse_results(raw_output, include_all)
            
            # Add to results with conversation ID
            for result in parsed_results:
                result["conversation_id"] = i + 1
                results.append(result)
                logger.info(f"Parsed result: {result}")
                
            logger.info(f"Completed conversation {i+1}, found {len(parsed_results)} relevant messages")
        
        # Save results to the fixed output path
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save CSV with all extracted messages
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["conversation_id", "speaker", "message", "crime_element"])
            writer.writeheader()
            writer.writerows(results)
        
        # Return the file response
        logger.info(f"Analysis completed in {time.time() - start_time:.2f}s. Results saved to {RESULTS_FILE}")
        
        return ResponseBody(FileResponse(path=str(RESULTS_FILE), file_type=FileType.CSV))
        
    except Exception as e:
        logger.error(f"Error analyzing conversations: {str(e)}")
        
        # Create error log file
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

# Add application metadata
server.add_app_metadata(
    name="Message Analyzer for Criminal Activity Extraction from Conversations",
    author="Satya Srujana Pilli, Shalom Jaison, Ashwini Ramesh Kumar",
    version="1.0.0",
    info="This application extracts and categorizes potential criminal activities from conversation text using a Gemma-2B ONNX model."
)

@server.app.route("/", methods=["GET"])
def root():
    """
    Root endpoint to verify that the server is running.
    
    Returns:
        str: A welcome message.
    """
    return "Welcome to the Message Analysis API!"

# Run the server
if __name__ == "__main__":
    print("Starting Message Analysis Server...")
    print(f"Model: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_NAME}")

    print(f"Available providers: {ort.get_available_providers()}")
    
    # Create results directory if it doesn't exist
    # RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start the server
    server.run(host="127.0.0.1", port=5000)
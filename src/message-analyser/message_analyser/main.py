import os
import time
import logging
from enum import Enum
from pathlib import Path
from typing import TypedDict
from functools import lru_cache
from rb.lib.ml_service import MLService
from rb.api.models import (
    ResponseBody,
    FileResponse,
    FileType,
    InputSchema,
    ParameterSchema,
    InputType,
    TextParameterDescriptor,
    TaskSchema,
    DirectoryInput,
    EnumParameterDescriptor,
    EnumVal,
    BatchFileInput,
)
import ollama
import typer
import traceback
from message_analyser.InputsHandler import InputsHandler
from message_analyser.OutputParser import OutputParser

project_root = Path(__file__).resolve().parent.parent.parent.parent


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemma-server")

APP_NAME = "message-analyser"

# Initialize Flask-ML Server
server = MLService(APP_NAME)


def load_model(model_name: str = "gemma"):
    if model_name == "mistral":
        return MistralOllamaInference()
    elif model_name == "gemma":
        return GemmaOllamaInference()
    else:
        raise ValueError(f"Unknown Model Name: {model_name}")


# Create a singleton instance of the inference engine
@lru_cache(maxsize=1)
def get_inference_engine(model_type):
    if model_type == "GEMMA3B":
        model_name = "gemma"
    else:
        model_name = "mistral"
    return load_model(model_name)


# Define the model type
class ModelType(str, Enum):
    GEMMA3 = "GEMMA3"
    MISTRAL7B = "MISTRAL7B"


class Usecases(str, Enum):
    Actus_Reus_analysis = "1"
    Mens_Rea_analysis = "2"
    Custom_prompt_analysis = "3"


class OutputType(str, Enum):
    csv = "csv"
    xlsx = "xlsx"
    pdf = "pdf"
    txt = "txt"


def map_outputfiletype_FileType(output_type):
    if output_type == "csv":
        return FileType.CSV
    elif output_type == "txt":
        return FileType.TEXT


# Define input and parameter types for Flask ML
class CrimeAnalysisInputs(TypedDict):
    """
    Defines the expected input structure for the crime analysis task.

    Attributes:
        input_file: Path to the CSV file containing conversations.
    """

    input_file: BatchFileInput
    output_file: DirectoryInput


class CrimeAnalysisParameters(TypedDict):
    """
    Defines the parameters used for crime analysis.

    Attributes:
        include_all_messages: Whether to include all messages or only those with crime elements.
    """

    model_name: str
    usecase: str
    output_type: str
    usecase3: str


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
        input_type=InputType.BATCHFILE,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )

    output_type = ParameterSchema(
        key="output_type",
        label="Output type",
        subtitle="Desired output file type",
        value=EnumParameterDescriptor(
            enum_vals=[EnumVal(key=mt.value, label=mt.name) for mt in OutputType],
            default=OutputType.csv.value,
        ),
    )

    model_schema = ParameterSchema(
        key="model_name",
        label="Model to use for analysis",
        subtitle="Choose GEMMA3 or MISTRAL7B",
        value=EnumParameterDescriptor(
            enum_vals=[EnumVal(key=mt.value, label=mt.name) for mt in ModelType],
            default=ModelType.MISTRAL7B.value,
        ),
    )

    prompt_schema = ParameterSchema(
        key="usecase3",
        label="Custom prompt for message analysing",
        subtitle="Enter a custom prompt to analyse your input conversation(s)",
        value=TextParameterDescriptor(default=""),
    )

    usecase_schema = ParameterSchema(
        key="usecase",
        label="Usecases of the message analyser",
        subtitle="",
        value=EnumParameterDescriptor(
            enum_vals=[EnumVal(key=mt.value, label=mt.name) for mt in Usecases],
            default=Usecases.Custom_prompt_analysis.value,
        ),
    )

    return TaskSchema(
        inputs=[input_schema, output_schema],  # Only input is the CSV file
        parameters=[
            model_schema,
            usecase_schema,
            output_type,
            prompt_schema,
        ],  # Parameters for analysis
    )


# Add application metadata
server.add_app_metadata(
    plugin_name=APP_NAME,
    name="Message Analyzer for Criminal Activity Extraction from Conversations",
    author="Satya Srujana Pilli, Shalom Jaison, Ashwini Ramesh Kumar",
    version="1.0.0",
    info="This application extracts and categorizes potential criminal activities from conversation text using a Gemma-2B ONNX model.",
)


def build_prompt(conversation: str, use_case: str, custom_prompt: str = None) -> str:
    """.
    Build the prompt based on the use case and provided conversation.

    Args:
        conversation (str): The conversation text in [Message i] Name: Content format.
        use_case (int): 1 for Actus Reus, 2 for Mens Rea, 3 for Custom Prompt.
        custom_prompt (str, optional): The user-defined prompt for use case 3.

    Returns:
        str: Final prompt ready for model input.
    """

    if use_case == "1":
        crime_element = "Actus Reus"
        definition = """**Definitions:**

        * **Actus Reus (Guilty Act):** This refers to the physical act of committing a crime. It's the tangible, observable action that constitutes the criminal offense."""
        no_element_response = (
            "No. There is no element of Actus Reus in the conversation."
        )
        gold_example = """
        GOLD EXAMPLE:

        Conversation:
        [Message 1] Liam: I smashed the window and climbed inside.
        [Message 2] Rachel: I grabbed the jewelry and ran out.
        [Message 3] Emily: Did you see the new bakery opening downtown?

        Expected Output:
        Yes. Evidence:
        Actus Reus:
        [Message 1 - Name]: I smashed the window and climbed inside.
        [Message 2 - Name]: I grabbed the jewelry and ran out.
        """

    elif use_case == "2":
        crime_element = "Mens Rea"
        definition = """* **Mens Rea (Guilty Mind):** This refers to the mental state of the perpetrator at the time the crime was committed. It encompasses the intention, knowledge, or recklessness that the person had when performing the act. In essence, it's about proving that the person knew what they were doing was wrong."""
        no_element_response = "No. There is no element of Mens Rea in the conversation."
        gold_example = """
        GOLD EXAMPLE:

        Conversation:
        [Message 1] Liam: We should plan the robbery carefully.
        [Message 2] Rachel: I'll study the security system tonight.
        [Message 3] Emily: Are you coming to the party tomorrow?

        Expected Output:
        Yes. Evidence:
        Mens Rea:
        [Message 1 - Name]: We should plan the robbery carefully.
        [Message 2 - Name]: I'll study the security system tonight.
        """

    elif use_case == "3":
        crime_element = custom_prompt if custom_prompt else "Relevant Messages"
        definition = ""
        no_element_response = (
            "No. There is no message that matches the prompt in the given conversation."
        )
        gold_example = """
        GOLD EXAMPLE:

        USER INSTRUCTION:
        Find messages showing distrust or suspicion.

        Conversation:
        [Message 1] Liam: Are you sure this plan will work?
        [Message 2] Rachel: I don't trust him with the money.
        [Message 3] Emily: Can't wait for the vacation next week!

        Expected Output:
        Yes. Evidence:
        Distrust or Suspicion:
        [Message 1 - Name]: Are you sure this plan will work?
        [Message 2 - Name]: I don't trust him with the money.
        """
    else:
        raise ValueError(
            f"Invalid use_case: {use_case}. Must be 1 (Actus Reus), 2 (Mens Rea), or 3 (Custom Prompt)."
        )

    prompt = f"""
    You are a forensic conversation analyst specializing in detecting **{crime_element}** from chat conversations.

    Your job is to carefully read the conversation and extract only the messages that match the definition of {crime_element}.
    {definition}
    ---

    DO NOT:
    - Do NOT paraphrase, summarize, or explain.
    - Do NOT guess or assume hidden meanings.
    - Do NOT include observations, interpretations, or conclusions.
    - ONLY output if there is clear evidence matching {crime_element}.

    ---

    STRICT OUTPUT FORMAT:
    Yes. Evidence:
    {crime_element}:
    [Message 1 - Name]: <exact message text>
    [Message 2 - Name]: <exact message text>

    If no relevant messages are found, output exactly:
    {no_element_response}

    ---

    INCORRECT FORMATS (DO NOT DO THIS):
    - {crime_element}: They probably meant something criminal.
    - {crime_element}: Someone sounded suspicious.
    - Any summaries, bullet points, or assumptions.

    {gold_example}

    ---

    Now analyze the following conversation:

    {conversation}

    ---
    Extract your output below:
    """

    return prompt


def analyse(model_name, list_of_conversations, usecase, custom_prompt="") -> list:
    """
    Process the DataFrame using the inference engine to extract criminal activities,
    then write the results to a CSV file in the given results directory.

    :param df: The input DataFrame with a 'conversation' column.
    :param crime_elements: A string of comma-separated crime elements (e.g., "Actus Reus,Mens Rea").
    :param results_dir: The directory where the output CSV should be saved.
    :return: The Path to the created CSV file.
    """
    engine = get_inference_engine(model_name)

    list_of_raw_outputs = []

    for i in range(len(list_of_conversations)):
        logger.info(f"Processing conversation {i+1}/{len(list_of_conversations)}")
        print(f"Processing conversation {i+1}/{len(list_of_conversations)}")
        prompt = build_prompt(list_of_conversations[i], usecase, custom_prompt)
        model_raw_output = engine.predict(prompt)
        list_of_raw_outputs.append(model_raw_output)
        logger.info(f"Completed conversation {i+1}")
        print(f"Completed conversation {i+1}")

    return list_of_raw_outputs


class MistralOllamaInference:
    """
    Inference engine using Ollama with Mistral 7B for text generation
    """

    def __init__(self, model_name="mistral:7b-instruct"):
        self.model_name = model_name

    def predict(self, prompt: str) -> str:
        try:
            ensure_model_exists("mistral:7b-instruct")
            logger.info("Prompt: %s", prompt)
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}")
            return f"Error: {str(e)}"


class GemmaOllamaInference:
    """
    Inference engine using Ollama with Gemma3: 12 Billion Parameter for text generation
    """

    def __init__(self, model_name="gemma3:12b"):
        self.model_name = model_name

    def predict(self, prompt: str) -> str:
        try:
            logger.info("Prompt: %s", prompt)
            response = ollama.chat(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            # logger.error(f"Ollama generation error: {str(e)}")
            return f"Error: {str(e)}"


def analyze_conversations(
    inputs: CrimeAnalysisInputs, parameters: CrimeAnalysisParameters
) -> ResponseBody:
    """
    Process a CSV file containing conversations, extract criminal activities using the Ollama model,
    and save results to a CSV file.
    """
    start_time = time.time()
    temp_file = None
    RESULTS_DIR = Path(inputs["output_file"].path)

    try:

        handler = InputsHandler(inputs)

        list_of_conversations = handler.load_conversations()

        usecase = parameters.get("usecase", Usecases.Actus_Reus_analysis.value)
        output_dir = inputs["output_file"]
        model_name = parameters.get("model_name", ModelType.GEMMA3.value)
        output_type = parameters.get("output_type", OutputType.csv.value)
        prompt = parameters.get("usecase3", "")
        if prompt == "empty":
            prompt = ""
        print("completed loading")

        list_of_raw_outputs = analyse(
            model_name, list_of_conversations, usecase, prompt
        )
        outputs = OutputParser(output_dir, output_type)
        outputs.process_raw_output(list_of_raw_outputs)
        logger.info(f"Analysis completed in {time.time() - start_time:.2f}s.")

        output_base = Path(output_dir.path) / f"analysis_of_conversations.{output_type}"
        print(f"Results saved to: {output_base}\n")
        file_response = FileResponse(
            path=str(output_base), file_type=map_outputfiletype_FileType(output_type)
        )

        return ResponseBody(file_response)

    except Exception as e:
        logger.error(f"Error analyzing conversations: {str(e)}")
        error_file = (
            RESULTS_DIR / "error_log.txt" if RESULTS_DIR else Path("error_log.txt")
        )
        with open(error_file, "w") as f:
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nStack trace:\n")
            f.write(traceback.format_exc())
        return ResponseBody(FileResponse(path=str(error_file), file_type=FileType.TEXT))

    finally:
        # Clean up the temporary file.
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"Removed temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")


def cli_parser(arg: str) -> dict:
    """
    Parse a single string argument into the input dict.

    Expected format:
      "<in1.csv>;<in2.csv>;...,<output_dir>"

    Returns:
      {
        "input_file": BatchFileInput(files=[...]),
        "output_file": DirectoryInput(path=...)
      }
    """
    try:
        files_part, out_dir = arg.split(",", 1)
    except ValueError:
        raise ValueError("Expected: '<in1>;<in2>;... , <output_dir>'")

    from pathlib import Path
    from rb.api.models import FileInput

    # Convert to absolute paths if not already
    file_paths = [p.strip() for p in files_part.split(";") if p.strip()]
    abs_paths = []
    for p in file_paths:
        path = Path(p)
        if not path.is_absolute():
            # If the file doesn't exist as a relative path from CWD,
            # try as relative path from project root
            if not Path.cwd().joinpath(path).exists():
                path = project_root.joinpath(path)
            else:
                path = Path.cwd().joinpath(path)
        abs_paths.append(str(path))

    # Log the paths for debugging
    logger.info(f"Input files: {abs_paths}")
    logger.info(f"Output directory: {out_dir}")

    # Create FileInput objects with absolute paths
    file_inputs = [FileInput(path=p) for p in abs_paths]

    return {
        "input_file": BatchFileInput(files=file_inputs),
        "output_file": DirectoryInput(path=str(Path(out_dir).absolute())),
    }


def param_parser(arg: str) -> dict:
    """
    Parse a single string argument into parameters.

    Expected format:
      "<model_name>,<output_type>,<usecase>,<custom_prompt?>"

    - model_name    e.g. GEMMA3 or MISTRAL7B
    - output_type   csv | xlsx | txt | pdf
    - usecase       1 | 2 | 3
    - custom_prompt only required if usecase==3 (can contain commas)

    Returns:
      {
        "model_name": ...,
        "output_type": ...,
        "usecase": ...,
        "usecase3": ...,
      }
    """
    parts = [p.strip() for p in arg.split(",", 3)]
    if len(parts) < 3:
        raise ValueError(
            "Expected at least 3 tokens: '<model>,<out_type>,<usecase>[,<prompt>]'"
        )
    model_name, output_type, usecase = parts[:3]
    custom = parts[3] if len(parts) == 4 else ""
    return {
        "model_name": model_name,
        "output_type": output_type,
        "usecase": usecase,
        "usecase3": custom,
    }


server.add_ml_service(
    rule="/analyze",
    ml_function=analyze_conversations,
    inputs_cli_parser=typer.Argument(
        parser=cli_parser,
        help="Comma-separated paths: <input_file_path>,<output_directory_path>",
    ),
    parameters_cli_parser=typer.Argument(
        parser=param_parser,
        help="Comma-separated crime elements (e.g., 'Actus Reus,Mens Rea'), and model type (e.g., 'GEMMA3').",
    ),
    short_title="Criminal Activity Extraction",
    order=0,
    task_schema_func=create_crime_analysis_task_schema,
)


def ensure_model_exists(model_name) -> None:
    response = ollama.pull(model_name)
    if response.status != "success":
        logger.exception(f"Failed to pull model '{model_name}': {response.error}")
        raise RuntimeError(f"Failed to pull model 'mistral:7b-instruct': {response}")


app = server.app

# Run the typer app
if __name__ == "__main__":
    logger.info("Starting Message Analysis Server...")
    app()

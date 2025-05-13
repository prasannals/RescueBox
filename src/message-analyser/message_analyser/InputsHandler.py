import logging
import tempfile
import shutil
import pdfplumber
import pandas as pd
from pathlib import Path
import re
import os

logger = logging.getLogger("file-loader")


class InputsHandler:
    """
    Handles single or batch file inputs for crime analysis.
    Supports CSV, TXT, PDF, and XLSX files and returns a flat list of conversations.
    """

    def __init__(self, inputs: dict):
        self.inputs = inputs

        if "input_file" in inputs:
            # Handle different types of input
            if hasattr(inputs["input_file"], "files"):
                # BatchFileInput object
                self.file_inputs = inputs["input_file"].files
            elif (
                isinstance(inputs["input_file"], dict)
                and "files" in inputs["input_file"]
            ):
                # Dictionary with files key
                self.file_inputs = inputs["input_file"]["files"]
            else:
                # Single file input
                self.file_inputs = [inputs["input_file"]]
        elif "input_files" in inputs:
            batch = inputs["input_files"]
            self.file_inputs = batch.files
        else:
            raise KeyError("Expected 'input_files' or 'input_file' in inputs")

    def _save_input_to_tempfile(self, file_input) -> Path:
        # figure out the original filename or path so we can keep its suffix
        if hasattr(file_input, "path"):
            original = file_input.path
        elif isinstance(file_input, dict) and "path" in file_input:
            original = file_input["path"]
        elif hasattr(file_input, "filename"):
            original = file_input.filename
        else:
            logger.error("No path or filename found in file input")
            original = ""

        ext = Path(original).suffix

        # create the temp file *with* the right suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = Path(temp.name)

        try:
            if hasattr(file_input, "file"):
                content = file_input.file.read()
                temp.write(content)
                temp.flush()
            elif hasattr(file_input, "path"):
                path_str = str(file_input.path)
                if os.path.exists(path_str):
                    shutil.copyfile(path_str, temp_path)
                else:
                    logger.error(f"Path does not exist: {path_str}")
                    raise ValueError(f"File not found: {path_str}")
            elif isinstance(file_input, dict) and "path" in file_input:
                path_str = file_input["path"]
                if os.path.exists(path_str):
                    shutil.copyfile(path_str, temp_path)
                else:
                    logger.error(f"Path does not exist: {path_str}")
                    raise ValueError(f"File not found: {path_str}")
            else:
                logger.error("Cannot extract content from file input")
                raise ValueError("Cannot extract file content")
        finally:
            temp.close()

        return temp_path

    def _extract_conversations(self, path: Path) -> list:
        ext = path.suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(path, header=0)
        elif ext == ".xlsx":
            df = pd.read_excel(path, header=0)
        else:
            df = None

        if df is not None:
            # Check if 'conversation' column exists
            if "conversation" in df.columns:
                col = df["conversation"]
            else:
                # Fallback to first column with warning
                if df.shape[1] > 1:
                    logger.warning(
                        f"Multiple columns in {path.name}, using first column only"
                    )
                col = df.iloc[:, 0]
            return col.dropna().astype(str).tolist()

        if ext == ".txt":
            text = path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            with pdfplumber.open(path) as pdf:
                pages = [
                    page.extract_text() for page in pdf.pages if page.extract_text()
                ]
            text = "\n".join(pages)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        blocks = re.split(r"\n-{3,}\n", text)
        return [blk.strip() for blk in blocks if blk.strip()]

    def load_conversations(self) -> list:
        """
        Extracts all conversations from the input files and returns a flat list.

        Returns:
            List[str]: All extracted conversation lines.
        """
        all_conversations = []
        for file_input in self.file_inputs:
            temp_path = self._save_input_to_tempfile(file_input)

            try:
                conversations = self._extract_conversations(temp_path)
                all_conversations.extend(conversations)
            except Exception as e:
                logger.error(f"Failed to process {temp_path}: {e}")
                raise ValueError(f"Could not process file {file_input}: {e}")
            finally:
                try:
                    temp_path.unlink()
                except Exception:
                    pass

        return all_conversations

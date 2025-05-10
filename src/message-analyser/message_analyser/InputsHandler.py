import logging
import tempfile
import shutil
import pdfplumber
import pandas as pd
from pathlib import Path
from rb.api.models import (
    FileInput,
    BatchFileInput,
)
import re

logger = logging.getLogger("file-loader")


class InputsHandler:
    """
    Handles single or batch file inputs for crime analysis.
    Supports CSV, TXT, PDF, and XLSX files and returns a flat list of conversations.
    """

    def __init__(self, inputs: dict):
        self.inputs = inputs
        if "input_files" in inputs:
            batch: BatchFileInput = inputs["input_files"]
            self.file_inputs = batch.files
        elif "input_file" in inputs:
            single: FileInput = inputs["input_file"]
            self.file_inputs = [single]
        else:
            raise KeyError("Expected 'input_files' or 'input_file' in inputs")

    def _save_input_to_tempfile(self, file_input) -> Path:
        # figure out the original filename or path so we can keep its suffix
        if hasattr(file_input, "path"):
            original = file_input.path
        elif hasattr(file_input, "filename"):
            original = file_input.filename
        else:
            original = ""
        ext = Path(original).suffix

        # create the temp file *with* the right suffix
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = Path(temp.name)
        try:
            if hasattr(file_input, "file"):
                logger.info("Reading from file_input.file")
                content = file_input.file.read()
                temp.write(content)
                temp.flush()
            elif hasattr(file_input, "path"):
                logger.info(f"Copying from disk path: {file_input.path}")
                shutil.copyfile(file_input.path, temp_path)
            else:
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
                logger.info(f"Extracting from file: {temp_path}")
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

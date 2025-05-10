import re
from typing import List, Dict
from pathlib import Path
import pandas as pd
from fpdf import FPDF
import unicodedata
from rb.api.models import DirectoryInput


class OutputParser:
    def __init__(self, output_dir: DirectoryInput, output_file_type: str):

        self.output_dir = output_dir
        self.output_file_type = output_file_type
        self.result_dir = Path(output_dir.path)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def process_raw_output(self, raw_output_list):

        results = []
        for i, raw_output in enumerate(raw_output_list):
            print(raw_output)
            curr_result = self.parse_results_grouped(
                raw_output, conversation_id=i + 1, chunk_id=i + 1
            )
            print(curr_result)
            results.append(curr_result)

        return self.save_to_file(results)

    def parse_results_grouped(
        self, model_output: str, conversation_id: int, chunk_id: int
    ) -> Dict[str, List[Dict]]:
        """
        Parses the model output and returns a dictionary grouping messages by crime element.
        """
        result = {"Answer": None, "Evidence": {"category": None, "message_text": None}}
        lines = model_output.strip().splitlines()
        if not lines:
            return result

        match_answer = re.match(
            r"^(Yes|No)\.(?:\s+Evidence:)?(.*)?$", lines[0], re.IGNORECASE
        )
        if not match_answer:
            return result

        answer = match_answer.group(1)
        evidence = match_answer.group(2).strip()

        if answer.lower() == "yes":
            evidence = []

            for line in lines[1:]:
                stripped_line = line.strip()
                if stripped_line:
                    category_match = re.match(r"^(.*?):$", stripped_line)
                    message_match = re.match(
                        r"^\[?Message\s+(\d+)\s*[-:]?\s*(.*?)\]?:?\s*(.*)",
                        stripped_line,
                        re.IGNORECASE,
                    )

                    if (
                        category_match
                    ):  # This essentially stores the prompt/category of the messages [Actus Reus, Mens Rea]
                        result["Evidence"]["category"] = category_match.group(1)
                    elif message_match:
                        evidence.append(stripped_line)
        else:
            evidence = [
                (
                    evidence
                    if evidence
                    else "\n".join([line.strip() for line in lines[1:] if line.strip()])
                )
            ]

        result["Answer"] = answer
        result["Evidence"]["message_text"] = evidence
        return result

    def clean_text_for_pdf(self, text: str) -> str:
        """
        Converts fancy unicode characters to closest ASCII equivalents.
        """
        if not isinstance(text, str):
            return text
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    def save_to_file(self, results: List[Dict]):
        rows = []
        for i, result in enumerate(results):
            conv_id = i + 1
            evidence = result["Evidence"]
            category = evidence["category"] if evidence else None
            messages = evidence["message_text"]

            if isinstance(messages, list):
                joined_messages = "\n".join(messages)
            else:
                joined_messages = messages
            rows.append(
                {
                    "conversation_id": conv_id,
                    "category": category,
                    "message_text": joined_messages,
                }
            )

        df = pd.DataFrame(rows)

        output_base = self.result_dir / "analysis_of_conversations"
        if self.output_file_type.lower() == "csv":
            df.to_csv(output_base.with_suffix(".csv"), index=False)
        elif self.output_file_type.lower() == "xlsx":
            df.to_excel(output_base.with_suffix(".xlsx"), index=False)
        elif self.output_file_type.lower() == "txt":
            with open(output_base.with_suffix(".txt"), "w", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    f.write(
                        f"{row['conversation_id']} | {row['category']} | {row['message_text']}\n"
                    )
        elif self.output_file_type.lower() == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for _, row in df.iterrows():
                self.clean_text_for_pdf(row["message_text"])
                pdf.multi_cell(
                    0,
                    10,
                    f"{row['conversation_id']} | {row['category']} | {row['message_text']}\n",
                )
            pdf.output(str(output_base.with_suffix(".pdf")))
        else:
            raise ValueError(f"Unsupported file type: {self.output_file_type}")

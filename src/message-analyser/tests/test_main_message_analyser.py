from pathlib import Path
from rb.api.models import AppMetadata
from message_analyser.main import (
    app as cli_app,
    APP_NAME,
    create_crime_analysis_task_schema,
    ModelType,
    OutputType,
    Usecases,
)
from rb.lib.common_tests import RBAppTest
from unittest.mock import patch


class TestMessageAnalyzer(RBAppTest):
    def setup_method(self):
        self.set_app(cli_app, APP_NAME)

    def get_metadata(self):
        return AppMetadata(
            name="Message Analyzer for Criminal Activity Extraction from Conversations",
            author="Satya Srujana Pilli, Shalom Jaison, Ashwini Ramesh Kumar",
            version="1.0.0",
            info="This application extracts and categorizes potential criminal activities from conversation text using a Gemma-2B ONNX model.",
            plugin_name=APP_NAME,
        )

    def get_all_ml_services(self):
        return [
            (
                0,
                "analyze",
                "Criminal Activity Extraction",
                create_crime_analysis_task_schema(),
            ),
        ]

    @patch("message_analyser.main.ensure_model_exists")
    def test_negative_test(self, ensure_model_exists_mock):
        analyze_api = f"/{APP_NAME}/analyze"
        bad_file = Path.cwd() / "src" / "message-analyser" / "tests" / "bad_test.csv"
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"
        # single bad input; no semicolon needed
        combined_arg = f"{bad_file},{results_dir}"
        # CLI params: MODEL,OUTPUT,USECASE
        params = f"{ModelType.GEMMA3.value},{OutputType.csv.value},{Usecases.Actus_Reus_analysis.value}"

        result = self.runner.invoke(cli_app, [analyze_api, combined_arg, params])

        # we expect it to error out on a bad file
        assert "Error analyzing" in result.stdout or result.exit_code != 0

    @patch("message_analyser.main.ensure_model_exists")
    def test_cli_analyze_command(self, ensure_model_exists_mock):
        analyze_cmd = "analyze"
        test_csv = (
            Path.cwd() / "src" / "message-analyser" / "tests" / "test_conversations.csv"
        )
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"
        combined_arg = f"{test_csv},{results_dir}"
        params = (
            f"{ModelType.GEMMA3.value},"
            f"{OutputType.csv.value},"
            f"{Usecases.Actus_Reus_analysis.value}"
        )

        # Prepare directories
        results_dir.mkdir(parents=True, exist_ok=True)

        mocked_csv_file = results_dir / "mocked_output.csv"
        mocked_csv_file.write_text("Mocked Analysis")

        # Patch write_results_to_csv to return the mock CSV file's path.
        with patch("message_analyser.main.analyse", return_value=str(mocked_csv_file)):
            self.runner.invoke(cli_app, [analyze_cmd, combined_arg, params])

    @patch("message_analyser.main.ensure_model_exists")
    def test_api_analyze_command(self, ensure_model_exists_mock):
        analyze_route = "/analyze"
        test_csv = (
            Path.cwd() / "src" / "message-analyser" / "tests" / "test_conversations.csv"
        )
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"

        # Prepare directories and expected output
        results_dir.mkdir(parents=True, exist_ok=True)
        expected_output = (
            results_dir / f"analysis_of_conversations.{OutputType.csv.value}"
        )

        # Build the JSON payload with the new parameter keys
        input_json = {
            "inputs": {
                # These keys now match your TypedDict: input_files & output_dir
                "input_files": {"paths": [str(test_csv)]},
                "output_dir": {"path": str(results_dir)},
            },
            "parameters": {
                "model_name": ModelType.GEMMA3.value,
                "output_type": OutputType.csv.value,
                "usecase": Usecases.Actus_Reus_analysis.value,
                "usecase3": "",
            },
        }

        with patch("message_analyser.main.analyse", return_value=["dummy"]), patch(
            "message_analyser.main.OutputParser.process_raw_output"
        ) as mock_parse:

            def _fake_parse(self, raws):
                expected_output.write_text("Mocked Analysis")

            mock_parse.side_effect = _fake_parse

            self.client.post(analyze_route, json=input_json)

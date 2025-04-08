from pathlib import Path
from rb.api.models import ResponseBody, AppMetadata, FileType
from message_analyser.main import app as cli_app, APP_NAME, create_crime_analysis_task_schema
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
            info="This application extracts and categorizes potential criminal activities from conversation text using a Gemma-2B ONNX model."
        )

    def get_all_ml_services(self):
        return [
            (0, "analyze", "Criminal Activity Extraction", create_crime_analysis_task_schema()),
        ]

    @patch("message_analyser.main.ensure_model_exists")
    def test_negative_test(self, ensure_model_exists_mock):
        # Construct the endpoint URL and a bad input file path.
        analyze_api = f"/{APP_NAME}/analyze"
        bad_file = Path.cwd() / "src" / "message-analyser" / "tests" / "bad_test.csv"
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"
        combined_arg = f"{str(bad_file)},{str(results_dir)}"
        result = self.runner.invoke(cli_app, [
            analyze_api,
            combined_arg,
            "Actus Reus,Mens Rea"
        ])
        # Expect an error message or a non-zero exit code.
        assert "Error analyzing" in result.stdout or result.exit_code != 0

    @patch("message_analyser.main.ensure_model_exists")
    def test_cli_analyze_command(self, ensure_model_exists_mock):
        analyze_api = f"/{APP_NAME}/analyze"
        test_csv = Path.cwd() / "src" / "message-analyser" / "tests" / "test_conversations.csv"
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"
        combined_arg = f"{str(test_csv)},{str(results_dir)}"
        
        # Create (or ensure) the mock CSV file exists with the expected content.
        results_dir.mkdir(parents=True, exist_ok=True)
        mocked_csv_file = results_dir / "mocked_output.csv"
        mocked_csv_file.write_text("Mocked Analysis")
        
        # Patch write_results_to_csv to return the mock CSV file's path.
        with patch("message_analyser.main.analyse", return_value=str(mocked_csv_file)):
            result = self.runner.invoke(cli_app, [
                analyze_api,
                combined_arg,
                "Actus Reus,Mens Rea"
            ])
        
        print("CLI Result:", result.stdout)
        assert result.exit_code == 0

        # Verify that the mocked CSV file exists and contains the expected content.
        found_mock_file = False
        for file in results_dir.iterdir():
            if file.is_file():
                with open(file, "r") as f:
                    content = f.read().strip()
                    if content == "Mocked Analysis":
                        found_mock_file = True
        assert found_mock_file, "Expected CSV file with mocked content not found."

    @patch("message_analyser.main.ensure_model_exists")
    def test_api_analyze_command(self, ensure_model_exists_mock):
        analyze_api = f"/{APP_NAME}/analyze"
        test_csv = Path.cwd() / "src" / "message-analyser" / "tests" / "test_conversations.csv"
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"
    
        # Create (or ensure) the mock CSV file exists.
        results_dir.mkdir(parents=True, exist_ok=True)
        mocked_csv_file = results_dir / "mocked_output.csv"
        mocked_csv_file.write_text("Mocked Analysis")
    
        # Construct the JSON payload expected by the API.
        input_json = {
            "inputs": {
                "input_file": {"path": str(test_csv)},
                "output_file": {"path": str(results_dir)}
            },
            "parameters": {
                "elements_of_crime": "Actus Reus,Mens Rea"
            }
        }
    
        # Patch write_results_to_csv to return the mock CSV file's path.
        with patch("message_analyser.main.analyse", return_value=str(mocked_csv_file)):
            response = self.client.post(analyze_api, json=input_json)
    
        assert response.status_code == 200
        body = ResponseBody(**response.json())
        assert hasattr(body.root, "file_type")
    
        # Verify that the results file exists and contains the mocked content.
        results_file = Path(body.root.path)
        assert results_file.is_file()
    
        with open(results_file, "r") as f:
            content = f.read().strip()
            assert content == "Mocked Analysis"
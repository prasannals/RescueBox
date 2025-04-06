from pathlib import Path

from rb.api.models import ResponseBody, AppMetadata, FileType
from message_analyser.main import app as cli_app, APP_NAME, create_crime_analysis_task_schema
from rb.lib.common_tests import RBAppTest


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

    def test_negative_test(self):
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

    def test_cli_analyze_command(self, caplog):
        with caplog.at_level("INFO"):
            analyze_api = f"/{APP_NAME}/analyze"
            test_csv = Path.cwd() / "src" / "message-analyser" / "tests" / "test_conversations.csv"
            results_dir = Path.cwd() / "src" / "message-analyser" / "results"
            # Combine the two paths into a single comma-separated argument as expected by your cli_parser
            combined_arg = f"{str(test_csv)},{str(results_dir)}"
            result = self.runner.invoke(cli_app, [
                analyze_api,
                combined_arg,
                "Actus Reus,Mens Rea"
            ])
            # Ensure the CLI command completes successfully.
            assert result.exit_code == 0
            # Check that a log message indicates that analysis has completed.
            assert any("Analysis completed" in message for message in caplog.messages)

    def test_api_analyze_command(self):
        analyze_api = f"/{APP_NAME}/analyze"
        test_csv = Path.cwd() / "src" / "message-analyser" / "tests" / "test_conversations.csv"
        results_dir = Path.cwd() / "src" / "message-analyser" / "results"

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
        response = self.client.post(analyze_api, json=input_json)
        assert response.status_code == 200

        body = ResponseBody(**response.json())
        # Assuming the response returns a FileResponse with file_type and path in a 'root' attribute.
        # Adjust the attribute access if your response structure differs.
        assert hasattr(body.root, "file_type")
        assert body.root.file_type == FileType.CSV

        # Verify that the results file was created.
        results_file = Path(body.root.path)
        assert results_file.is_file()
"""Test cases for the __main__ module."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from laptrack import __main__


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


def test_main_succeeds(runner: CliRunner, tmp_path: Path) -> None:
    """It exits with a status code of zero and writes the output file."""
    csv_path = Path("docs/examples/sample_data.csv")
    output = tmp_path / "out.geff"
    result = runner.invoke(
        __main__.main, ["--csv_path", str(csv_path), "--output", str(output)]
    )
    assert result.exit_code == 0
    assert output.exists()

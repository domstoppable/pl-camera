from pathlib import Path

import pytest

ROOT_PATH = Path(__file__).parent.parent
TEST_DATA_PATH = ROOT_PATH / "tests" / "data"


@pytest.fixture
def test_data_path() -> Path:
    return TEST_DATA_PATH

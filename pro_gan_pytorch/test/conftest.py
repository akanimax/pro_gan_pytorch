from pathlib import Path

# noinspection PyPackageRequirements
import pytest


@pytest.fixture
def test_data_path() -> Path:
    return Path("/home/animesh/work/data/3d_scenes/forest_synthetic_struct/images")

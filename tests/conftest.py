import shutil
from pathlib import Path
from typing import Generator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig
from pyprojroot import here


@pytest.fixture(scope="session", autouse=True)
def cfg() -> Generator[DictConfig, None, None]:
    with initialize(version_base="1.3", config_path=str("../config/")):
        cfg = compose(config_name="integration_test_config")
        yield cfg


def pytest_sessionstart(session):
    if not Path(here() / "logs/test_logs").exists():
        (here() / "logs/test_logs").mkdir(parents=True)


def pytest_sessionfinish(session, exitstatus):
    if (here() / "logs/test_logs").exists():
        shutil.rmtree(here() / "logs/test_logs")

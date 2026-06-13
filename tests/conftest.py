import os
import pytest

@pytest.fixture(autouse=True)
def change_test_dir(monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

"""Streamlit evidence-room smoke contract."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard():
    return AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)


def test_dashboard_renders_without_exceptions(dashboard):
    assert not dashboard.exception
    assert dashboard.title[0].value == "Chip Test Policy Control Room"


def test_dashboard_uses_canonical_and_live_metrics(dashboard):
    values = [metric.value for metric in dashboard.metric]

    assert "13.524%" in values
    assert "6 / 600" in values
    assert "13.500%" in values
    assert "450" in values
    assert "50" in values


def test_dashboard_discloses_public_synthetic_scope(dashboard):
    captions = " ".join(element.value for element in dashboard.caption)

    assert "Public synthetic reconstruction" in captions
    assert "no production or confidential data" in captions
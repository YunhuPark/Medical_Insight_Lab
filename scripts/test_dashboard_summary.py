"""Exercise the dashboard a visitor sees, including its published PSM result."""
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_uses_corrected_full_sample_psm_result():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard.py"))
    app.run(timeout=120)
    assert not app.exception
    displayed = "\n".join(item.value for item in app.markdown)
    assert "0.65배" in displayed
    assert "p=0.036" in displayed
    assert "0.63배" not in displayed
    assert "p=0.019" not in displayed
    assert "전체 데이터" in displayed

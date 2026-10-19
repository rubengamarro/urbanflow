from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_displays_api_failure_without_crash(monkeypatch):
    import httpx

    def fail(*args, **kwargs):
        raise httpx.ConnectError("test API unavailable")

    monkeypatch.setattr(httpx, "get", fail)
    dashboard = Path(__file__).parents[1] / "apps/dashboard.py"
    app = AppTest.from_file(str(dashboard)).run(timeout=30)
    assert not app.exception
    assert "API not ready" in app.error[0].value

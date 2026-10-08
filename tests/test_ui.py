from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).resolve().parent.parent / "streamlit_ui.py")


def test_streamlit_app_starts_and_initialises_offline(monkeypatch):
    monkeypatch.setenv("MULTI_AGENT_OFFLINE", "1")
    at = AppTest.from_file(UI, default_timeout=60).run()
    assert not at.exception
    init = [b for b in at.sidebar.button if "Initialize" in b.label][0]
    init.click().run()
    assert not at.exception
    assert any("System initialized" in s.value for s in at.sidebar.success)

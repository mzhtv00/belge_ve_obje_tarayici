from pathlib import Path

from app import exports


def test_slicer_launch_uses_argument_list_without_shell(tmp_path, monkeypatch):
    executable = tmp_path / "cura.exe"
    executable.write_bytes(b"")
    model = tmp_path / "part.stl"
    model.write_bytes(b"")
    calls = []

    monkeypatch.setenv("SLICER_EXECUTABLE", str(executable))
    monkeypatch.setattr(exports.subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = exports.open_in_slicer(model)

    assert result == str(executable.resolve())
    assert calls == [(([str(executable.resolve()), str(model.resolve())],), {"shell": False})]

import os
import shutil
import subprocess
from pathlib import Path

import webbrowser


TINKERCAD_URL = "https://www.tinkercad.com/"
SLICER_COMMANDS = ("UltiMaker-Cura", "cura", "PrusaSlicer", "prusa-slicer", "slic3r")


class SlicerNotFound(RuntimeError):
    pass


def find_slicer() -> str:
    configured = os.environ.get("SLICER_EXECUTABLE")
    if configured:
        executable = Path(configured).expanduser()
        if executable.is_file():
            return str(executable.resolve())
        raise SlicerNotFound("SLICER_EXECUTABLE is set, but the file does not exist.")
    for command in SLICER_COMMANDS:
        executable = shutil.which(command)
        if executable:
            return executable
    raise SlicerNotFound(
        "No supported slicer was found on PATH. Set SLICER_EXECUTABLE to Cura or PrusaSlicer."
    )


def open_in_slicer(model_path: Path) -> str:
    executable = find_slicer()
    subprocess.Popen([executable, str(model_path.resolve())], shell=False)
    return executable


def open_tinkercad() -> None:
    if not webbrowser.open(TINKERCAD_URL, new=2):
        raise RuntimeError("The default browser could not open Tinkercad.")

import threading
import time
import webbrowser

import uvicorn

from app.config import Settings


def open_dashboard() -> None:
    time.sleep(1.2)
    settings = Settings.from_env()
    webbrowser.open(f"http://127.0.0.1:{settings.port}/")


if __name__ == "__main__":
    settings = Settings.from_env()
    threading.Thread(target=open_dashboard, daemon=True).start()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        ws_max_size=256 * 1024,
    )

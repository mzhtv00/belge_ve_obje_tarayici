import os
import socket
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    host: str = "0.0.0.0"
    port: int = 8000
    max_image_bytes: int = 12 * 1024 * 1024
    max_model_bytes: int = 100 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    worker_count: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("SCANNER_DATA_DIR", "data")).expanduser(),
            host=os.environ.get("SCANNER_HOST", "0.0.0.0"),
            port=int(os.environ.get("SCANNER_PORT", "8000")),
            worker_count=max(1, int(os.environ.get("SCANNER_WORKERS", "2"))),
        )

    def prepare(self) -> None:
        (self.data_dir / "captures").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "thumbnails").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "models").mkdir(parents=True, exist_ok=True)


def local_ip() -> str:
    """Return the address used for LAN routing without sending network traffic."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()

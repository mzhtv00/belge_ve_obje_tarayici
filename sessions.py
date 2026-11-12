import secrets
import uuid
from dataclasses import dataclass
from hmac import compare_digest


VALID_MODES = {"2d", "3d"}


@dataclass
class ScanSession:
    id: str
    token: str
    mode: str = "2d"


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ScanSession] = {}

    def create(self) -> ScanSession:
        session = ScanSession(id=uuid.uuid4().hex, token=secrets.token_urlsafe(32))
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> ScanSession | None:
        return self._sessions.get(session_id)

    def authorize(self, session_id: str, token: str | None) -> ScanSession | None:
        session = self.get(session_id)
        if session is None or token is None or not compare_digest(session.token, token):
            return None
        return session

    def set_mode(self, session_id: str, mode: str) -> ScanSession | None:
        session = self.get(session_id)
        if session is None or mode not in VALID_MODES:
            return None
        session.mode = mode
        return session

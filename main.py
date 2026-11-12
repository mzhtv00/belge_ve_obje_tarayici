import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import qrcode
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import Settings, local_ip
from app.exports import TINKERCAD_URL, SlicerNotFound, open_in_slicer, open_tinkercad
from app.pipeline import CapturePipeline, EventHub
from app.sessions import SessionManager, VALID_MODES
from app.storage import CaptureStore, InvalidImage, InvalidModel, ModelStore, validate_stl

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


class ModeRequest(BaseModel):
    mode: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.prepare()
    sessions = SessionManager()
    hub = EventHub()
    captures = CaptureStore(settings.data_dir, settings.max_image_pixels)
    models = ModelStore(settings.data_dir)
    pipeline = CapturePipeline(captures, hub, settings.worker_count)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await pipeline.start()
        try:
            yield
        finally:
            await pipeline.stop()

    app = FastAPI(title="Mobile Scan and CAD Handoff", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions
    app.state.hub = hub
    app.state.captures = captures
    app.state.models = models
    app.state.pipeline = pipeline
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/mobile/{session_id}")
    async def mobile_page(session_id: str) -> FileResponse:
        if sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="Scan session not found.")
        return FileResponse(STATIC_DIR / "mobile.html")

    @app.post("/api/sessions")
    async def create_session() -> dict[str, Any]:
        session = sessions.create()
        base_url = f"http://{local_ip()}:{settings.port}"
        mobile_url = f"{base_url}/mobile/{session.id}?token={quote(session.token)}"
        return {
            "id": session.id,
            "token": session.token,
            "mode": session.mode,
            "mobile_url": mobile_url,
            "qr_url": f"/api/sessions/{session.id}/qr.png",
        }

    @app.get("/api/sessions/{session_id}/qr.png")
    async def session_qr(session_id: str) -> StreamingResponse:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Scan session not found.")
        image = qrcode.make(f"http://{local_ip()}:{settings.port}/mobile/{session.id}?token={quote(session.token)}")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        stream.seek(0)
        return StreamingResponse(stream, media_type="image/png")

    def require_session(session_id: str, token: str | None) -> Any:
        session = sessions.authorize(session_id, token)
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid or expired scan session token.")
        return session

    @app.patch("/api/sessions/{session_id}/mode")
    async def update_mode(session_id: str, request: ModeRequest, token: str | None = Query(default=None)) -> dict[str, str]:
        require_session(session_id, token)
        if request.mode not in VALID_MODES:
            raise HTTPException(status_code=422, detail="Mode must be '2d' or '3d'.")
        session = sessions.set_mode(session_id, request.mode)
        await hub.publish(session_id, {"type": "session.mode", "mode": session.mode})
        return {"mode": session.mode}

    @app.get("/api/captures")
    async def list_captures(
        session_id: str,
        token: str,
        q: str = "",
        mode: str = "all",
    ) -> dict[str, list[dict[str, Any]]]:
        require_session(session_id, token)
        term = q.strip().casefold()
        results = []
        for record in captures.list_captures():
            if record.get("session_id") != session_id:
                continue
            if mode != "all" and record.get("mode") != mode:
                continue
            if term and term not in " ".join(
                (record.get("filename", ""), record.get("captured_at", ""), record.get("mode", ""))
            ).casefold():
                continue
            results.append({
                **record,
                "image_url": f"/api/captures/{record['id']}/image",
                "thumbnail_url": f"/api/captures/{record['id']}/thumbnail",
            })
        return {"captures": results}

    def require_authorized_query(session_id: str, token: str | None) -> None:
        require_session(session_id, token)

    @app.get("/api/captures/{capture_id}/image")
    async def capture_image(capture_id: str, session_id: str, token: str | None = Query(default=None)) -> FileResponse:
        require_authorized_query(session_id, token)
        record = captures.get_capture(capture_id)
        path = captures.image_path(capture_id)
        if record is None or path is None or record.get("session_id") != session_id:
            raise HTTPException(status_code=404, detail="Capture not found.")
        return FileResponse(path, media_type=record["mime_type"])

    @app.get("/api/captures/{capture_id}/thumbnail")
    async def capture_thumbnail(capture_id: str, session_id: str, token: str | None = Query(default=None)) -> FileResponse:
        require_authorized_query(session_id, token)
        record = captures.get_capture(capture_id)
        path = captures.thumbnail_path(capture_id)
        if record is None or record.get("session_id") != session_id or path is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found.")
        return FileResponse(path, media_type="image/jpeg")

    async def read_limited(file: UploadFile, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await file.read(min(64 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit.")
            chunks.append(chunk)
        return b"".join(chunks)

    @app.post("/api/models")
    async def upload_model(
        session_id: str,
        token: str,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        require_session(session_id, token)
        if Path(file.filename or "").suffix.casefold() != ".stl":
            raise HTTPException(status_code=415, detail="Only .stl files are supported.")
        data = await read_limited(file, settings.max_model_bytes)
        try:
            validate_stl(data, settings.max_model_bytes)
        except InvalidModel as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"model": models.save_model(data, file.filename or "model.stl", session_id)}

    @app.get("/api/models")
    async def list_models(session_id: str, token: str) -> dict[str, list[dict[str, Any]]]:
        require_session(session_id, token)
        return {"models": models.list_models(session_id)}

    @app.get("/api/models/{model_id}/download")
    async def download_model(model_id: str, session_id: str, token: str | None = Query(default=None)) -> FileResponse:
        require_authorized_query(session_id, token)
        record = models.get_model_record(model_id, session_id)
        path = models.get_model(model_id, session_id)
        if record is None or path is None:
            raise HTTPException(status_code=404, detail="STL model not found.")
        return FileResponse(path, filename=record["original_name"], media_type="application/sla")

    @app.post("/api/models/{model_id}/open-slicer")
    async def slicer_export(model_id: str, session_id: str, token: str) -> dict[str, str]:
        require_session(session_id, token)
        path = models.get_model(model_id, session_id)
        if path is None:
            raise HTTPException(status_code=404, detail="STL model not found.")
        try:
            executable = open_in_slicer(path)
        except SlicerNotFound as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            logger.exception("Could not launch the configured slicer.")
            raise HTTPException(status_code=500, detail="The slicer could not be started.") from exc
        return {"message": "STL opened in slicer.", "slicer": Path(executable).name}

    @app.post("/api/models/{model_id}/open-tinkercad")
    async def tinkercad_export(model_id: str, session_id: str, token: str) -> dict[str, str]:
        require_session(session_id, token)
        if models.get_model(model_id, session_id) is None:
            raise HTTPException(status_code=404, detail="STL model not found.")
        try:
            await asyncio.to_thread(open_tinkercad)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "message": "Tinkercad opened. Download the STL and import it in a new Tinkercad design.",
            "url": TINKERCAD_URL,
        }

    @app.websocket("/ws/dashboard/{session_id}")
    async def dashboard_socket(websocket: WebSocket, session_id: str, token: str = Query(...)) -> None:
        if sessions.authorize(session_id, token) is None:
            await websocket.close(code=1008, reason="Invalid session token")
            return
        await hub.connect(session_id, websocket)
        try:
            await websocket.send_json({"type": "connected", "mode": sessions.get(session_id).mode})
            while True:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect(session_id, websocket)

    @app.websocket("/ws/mobile/{session_id}")
    async def mobile_socket(websocket: WebSocket, session_id: str, token: str = Query(...)) -> None:
        session = sessions.authorize(session_id, token)
        if session is None:
            await websocket.close(code=1008, reason="Invalid session token")
            return
        await hub.connect(session_id, websocket)
        upload: dict[str, Any] | None = None
        try:
            await websocket.send_json({"type": "connected"})
            await websocket.send_json({"type": "session.mode", "mode": session.mode})
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    try:
                        control = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "message": "Invalid upload message."})
                        continue
                    if not isinstance(control, dict):
                        await websocket.send_json({"type": "error", "message": "Invalid upload message."})
                        continue
                    if control.get("type") == "capture.start":
                        if upload is not None:
                            await websocket.send_json({"type": "error", "message": "Finish the current upload first."})
                            continue
                        mime_type = control.get("mime_type")
                        if not isinstance(mime_type, str) or mime_type not in {"image/jpeg", "image/png", "image/webp"}:
                            await websocket.send_json({"type": "error", "message": "Unsupported image format."})
                            continue
                        upload = {"mime_type": mime_type, "data": bytearray()}
                    elif control.get("type") == "capture.end" and upload is not None:
                        data = bytes(upload["data"])
                        try:
                            record = await asyncio.to_thread(
                                captures.save_capture,
                                data,
                                session_id,
                                session.mode,
                                settings.max_image_bytes,
                            )
                        except InvalidImage as exc:
                            await websocket.send_json({"type": "error", "message": str(exc)})
                        else:
                            await pipeline.enqueue(session_id, record["id"])
                            event = {"type": "capture.accepted", "capture": record}
                            await hub.publish(session_id, event)
                        upload = None
                    else:
                        await websocket.send_json({"type": "error", "message": "Unexpected upload message."})
                elif message.get("bytes") is not None:
                    chunk = message["bytes"]
                    if upload is None:
                        await websocket.send_json({"type": "error", "message": "Start an upload before sending image data."})
                        continue
                    if len(chunk) > 256 * 1024 or len(upload["data"]) + len(chunk) > settings.max_image_bytes:
                        upload = None
                        await websocket.send_json({"type": "error", "message": "Image exceeds the configured size limit."})
                        continue
                    upload["data"].extend(chunk)
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect(session_id, websocket)

    return app


app = create_app()

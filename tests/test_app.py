import io
import struct

from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.config import Settings
from app.exports import SlicerNotFound
from app.main import create_app


def jpeg_bytes() -> bytes:
    image = Image.new("RGB", (24, 16), color=(30, 180, 130))
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def binary_stl() -> bytes:
    triangle = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    return b"binary STL header".ljust(80, b"\x00") + (1).to_bytes(4, "little") + triangle


def test_session_qr_and_mode_authorization(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        response = client.post("/api/sessions")
        assert response.status_code == 200
        session = response.json()

        qr = client.get(session["qr_url"])
        assert qr.status_code == 200
        assert qr.headers["content-type"] == "image/png"
        assert qr.content.startswith(b"\x89PNG")

        invalid_auth = client.patch(
            f"/api/sessions/{session['id']}/mode?token=wrong",
            json={"mode": "3d"},
        )
        assert invalid_auth.status_code == 401

        mode = client.patch(
            f"/api/sessions/{session['id']}/mode?token={session['token']}",
            json={"mode": "3d"},
        )
        assert mode.status_code == 200
        assert mode.json() == {"mode": "3d"}


def test_dashboard_and_mobile_assets_are_served(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        dashboard_script = client.get("/static/dashboard.js")
        mobile_script = client.get("/static/mobile.js")
        assert dashboard_script.status_code == mobile_script.status_code == 200
        assert "import(\"three\")" in dashboard_script.text
        assert "capture.start" in mobile_script.text
        assert "capture.processed" in mobile_script.text


def test_mobile_websocket_stores_capture_and_broadcasts_events(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        session = client.post("/api/sessions").json()
        dashboard_url = f"/ws/dashboard/{session['id']}?token={session['token']}"
        mobile_url = f"/ws/mobile/{session['id']}?token={session['token']}"
        with client.websocket_connect(dashboard_url) as dashboard:
            assert dashboard.receive_json()["type"] == "connected"
            with client.websocket_connect(mobile_url) as mobile:
                assert mobile.receive_json()["type"] == "connected"
                assert mobile.receive_json()["type"] == "session.mode"
                mobile.send_json({"type": "capture.start", "mime_type": "image/jpeg"})
                mobile.send_bytes(jpeg_bytes())
                mobile.send_json({"type": "capture.end"})
                accepted = mobile.receive_json()
                assert accepted["type"] == "capture.accepted"
                assert accepted["capture"]["mode"] == "2d"
                processed = mobile.receive_json()
                assert processed["type"] == "capture.processed"

                events = [dashboard.receive_json(), dashboard.receive_json()]
                event_types = {event["type"] for event in events}
                assert event_types == {"capture.accepted", "capture.processed"}

        listing = client.get("/api/captures", params={
            "session_id": session["id"],
            "token": session["token"],
        })
        assert listing.status_code == 200
        capture = listing.json()["captures"][0]
        assert capture["status"] == "stored"
        assert capture["thumbnail"]
        image = client.get(capture["image_url"], params={
            "session_id": session["id"],
            "token": session["token"],
        })
        assert image.status_code == 200
        assert image.content.startswith(b"\xff\xd8")


def test_upload_requires_authorization_and_rejects_non_stl(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        session = client.post("/api/sessions").json()
        unauthorized = client.get("/api/models", params={"session_id": session["id"], "token": "wrong"})
        assert unauthorized.status_code == 401

        response = client.post(
            f"/api/models?session_id={session['id']}&token={session['token']}",
            files={"file": ("model.stl", b"not an STL model", "application/sla")},
        )
        assert response.status_code == 422


def test_model_upload_list_and_download_are_session_scoped(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        first = client.post("/api/sessions").json()
        second = client.post("/api/sessions").json()
        upload = client.post(
            "/api/models",
            params={"session_id": first["id"], "token": first["token"]},
            files={"file": ("sample model.stl", binary_stl(), "application/sla")},
        )
        assert upload.status_code == 200
        model = upload.json()["model"]

        first_models = client.get("/api/models", params={
            "session_id": first["id"],
            "token": first["token"],
        })
        second_models = client.get("/api/models", params={
            "session_id": second["id"],
            "token": second["token"],
        })
        assert [item["id"] for item in first_models.json()["models"]] == [model["id"]]
        assert second_models.json()["models"] == []

        own_download = client.get(
            f"/api/models/{model['id']}/download",
            params={"session_id": first["id"], "token": first["token"]},
        )
        other_download = client.get(
            f"/api/models/{model['id']}/download",
            params={"session_id": second["id"], "token": second["token"]},
        )
        assert own_download.status_code == 200
        assert own_download.content == binary_stl()
        assert "sample%20model.stl" in own_download.headers["content-disposition"]
        assert other_download.status_code == 404
        other_slicer = client.post(
            f"/api/models/{model['id']}/open-slicer",
            params={"session_id": second["id"], "token": second["token"]},
        )
        assert other_slicer.status_code == 404


def test_stl_upload_enforces_configured_file_size_limit(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, max_model_bytes=100))
    with TestClient(app) as client:
        session = client.post("/api/sessions").json()
        response = client.post(
            "/api/models",
            params={"session_id": session["id"], "token": session["token"]},
            files={"file": ("sample.stl", binary_stl(), "application/sla")},
        )
        assert response.status_code == 413


def test_model_handoff_routes_report_slicer_and_tinkercad_results(tmp_path, monkeypatch):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        session = client.post("/api/sessions").json()
        upload = client.post(
            "/api/models",
            params={"session_id": session["id"], "token": session["token"]},
            files={"file": ("sample.stl", binary_stl(), "application/sla")},
        )
        model_id = upload.json()["model"]["id"]
        monkeypatch.setattr(main, "open_tinkercad", lambda: None)
        tinkercad = client.post(
            f"/api/models/{model_id}/open-tinkercad",
            params={"session_id": session["id"], "token": session["token"]},
        )
        assert tinkercad.status_code == 200
        assert tinkercad.json()["url"] == "https://www.tinkercad.com/"

        monkeypatch.setattr(
            "app.exports.find_slicer",
            lambda: (_ for _ in ()).throw(SlicerNotFound("No supported slicer was found.")),
        )
        slicer = client.post(
            f"/api/models/{model_id}/open-slicer",
            params={"session_id": session["id"], "token": session["token"]},
        )
        assert slicer.status_code == 503
        assert slicer.json()["detail"] == "No supported slicer was found."


def test_malformed_websocket_control_messages_do_not_abort_connection(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        session = client.post("/api/sessions").json()
        with client.websocket_connect(
            f"/ws/mobile/{session['id']}?token={session['token']}"
        ) as mobile:
            assert mobile.receive_json()["type"] == "connected"
            assert mobile.receive_json()["type"] == "session.mode"
            mobile.send_json([])
            assert mobile.receive_json()["type"] == "error"
            mobile.send_json({"type": "capture.start", "mime_type": []})
            assert mobile.receive_json()["type"] == "error"
            mobile.send_json({"type": "capture.start", "mime_type": "image/jpeg"})
            mobile.send_bytes(jpeg_bytes())
            mobile.send_json({"type": "capture.end"})
            assert mobile.receive_json()["type"] == "capture.accepted"


def test_capture_listing_is_scoped_to_authorized_session(tmp_path):
    app = create_app(Settings(data_dir=tmp_path))
    with TestClient(app) as client:
        first = client.post("/api/sessions").json()
        second = client.post("/api/sessions").json()
        for session in (first, second):
            with client.websocket_connect(
                f"/ws/mobile/{session['id']}?token={session['token']}"
            ) as mobile:
                assert mobile.receive_json()["type"] == "connected"
                assert mobile.receive_json()["type"] == "session.mode"
                mobile.send_json({"type": "capture.start", "mime_type": "image/jpeg"})
                mobile.send_bytes(jpeg_bytes())
                mobile.send_json({"type": "capture.end"})
                assert mobile.receive_json()["type"] == "capture.accepted"

        first_list = client.get("/api/captures", params={
            "session_id": first["id"],
            "token": first["token"],
        })
        second_list = client.get("/api/captures", params={
            "session_id": second["id"],
            "token": second["token"],
        })
        assert first_list.status_code == second_list.status_code == 200
        assert len(first_list.json()["captures"]) == 1
        assert len(second_list.json()["captures"]) == 1
        assert first_list.json()["captures"][0]["session_id"] == first["id"]
        assert second_list.json()["captures"][0]["session_id"] == second["id"]

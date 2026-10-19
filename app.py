"""Giao diện web FaceID Lab: nhận diện qua webcam trình duyệt, đăng ký người mới, xem đánh giá.

    .venv\\Scripts\\python -m uvicorn app:app --port 8000
rồi mở http://localhost:8000
"""
import asyncio
import base64
import csv
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Log có tiếng Việt; khi stdout bị chuyển hướng trên Windows (cp1252) print sẽ ném lỗi.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from calibrate_threshold import collect_scores, rates_at
from face_runtime import create_face_app, normalized_embedding
from preprocessing import (
    EMBEDDINGS_FILE,
    PROTOTYPES_FILE,
    build_prototypes,
    face_passes_quality,
)


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
LOG_DIR = "logs"
LOG_COOLDOWN_SECONDS = 30
UNKNOWN = "Người lạ"
CALIBRATION_THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(11)]

# InsightFace không an toàn khi nhiều luồng gọi cùng lúc: mọi lần suy luận đi qua khóa này.
face_lock = threading.Lock()
face_app = create_face_app(allowed_modules=["detection", "recognition"])


class Gallery:
    """Embedding, prototype và số liệu đánh giá đang dùng; nạp lại sau mỗi lần đăng ký."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reload()

    def reload(self):
        with np.load(PROTOTYPES_FILE, allow_pickle=False) as model:
            labels = model["labels"].astype(str)
            prototypes = model["prototypes"].astype(np.float32)
            threshold = float(model["cosine_threshold"].item())

        embeddings = None
        if os.path.exists(EMBEDDINGS_FILE):
            with np.load(EMBEDDINGS_FILE, allow_pickle=False) as cache:
                embeddings = {
                    "signature": str(cache["signature"].item()),
                    "embeddings": cache["embeddings"].astype(np.float32),
                    "labels": cache["labels"].astype(str),
                    "video_ids": cache["video_ids"].astype(str),
                }

        with self.lock:
            self.labels, self.prototypes, self.threshold = labels, prototypes, threshold
            self.embeddings = embeddings
            self.report = self._build_report()

    def match(self, embedding):
        similarities = self.prototypes @ embedding
        best = int(np.argmax(similarities))
        return str(self.labels[best]), float(similarities[best])

    def _build_report(self):
        report = {"people": [], "calibration": [], "videos": 0, "frames": 0}
        nearest = {}
        if len(self.labels) > 1:
            similarities = self.prototypes @ self.prototypes.T
            np.fill_diagonal(similarities, -1)
            for i, label in enumerate(self.labels):
                j = int(np.argmax(similarities[i]))
                nearest[label] = (str(self.labels[j]), float(similarities[i, j]))

        data = self.embeddings
        for label in self.labels:
            videos = frames = 0
            if data is not None:
                mask = data["labels"] == label
                frames = int(mask.sum())
                videos = int(len(np.unique(data["video_ids"][mask])))
            other, score = nearest.get(label, ("", 0.0))
            report["people"].append(
                {"name": str(label), "videos": videos, "frames": frames, "nearest": other, "nearestScore": round(score, 3)}
            )

        if data is not None and len(np.unique(data["labels"])) > 1:
            report["videos"] = int(len(np.unique(data["video_ids"])))
            report["frames"] = int(len(data["labels"]))
            known, unknown = collect_scores(data["embeddings"], data["labels"], data["video_ids"])
            report["top1"] = float(np.mean(known["correct"])) if len(known["correct"]) else None
            for threshold in CALIBRATION_THRESHOLDS:
                rates = rates_at(threshold, known, unknown)
                report["calibration"].append(
                    {"t": threshold, "tar": round(float(rates["tar"]), 4), "far": round(float(rates["far_macro"]), 4)}
                )
        return report


gallery = Gallery()


class AttendanceLog:
    """Ghi logs/attendance_YYYY-MM-DD.csv; mỗi người chỉ ghi lại sau LOG_COOLDOWN_SECONDS."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_seen = {}
        self.recent = []

    def path(self, day=None):
        return os.path.join(LOG_DIR, f"attendance_{day or datetime.now():%Y-%m-%d}.csv")

    def record(self, name, score):
        now = time.time()
        with self.lock:
            if now - self.last_seen.get(name, 0) < LOG_COOLDOWN_SECONDS:
                return
            self.last_seen[name] = now
            stamp = datetime.now()
            os.makedirs(LOG_DIR, exist_ok=True)
            path = self.path(stamp)
            is_new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                if is_new:
                    writer.writerow(["time", "name", "score"])
                writer.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S"), name, f"{score:.3f}"])
            self.recent.insert(0, {"time": stamp.strftime("%H:%M:%S"), "name": name, "score": round(score, 2)})
            del self.recent[20:]


attendance = AttendanceLog()


def decode_image(data):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không giải mã được ảnh")
    return image


def recognize(frame, threshold, log_enabled):
    with face_lock:
        faces = face_app.get(frame)
    height, width = frame.shape[:2]
    results = []
    for face in faces:
        embedding = normalized_embedding(face)
        if embedding is None:
            continue
        best, score = gallery.match(embedding)
        known = score >= threshold
        x1, y1, x2, y2 = face.bbox.tolist()
        results.append({
            "x": max(0.0, x1 / width), "y": max(0.0, y1 / height),
            "w": (x2 - x1) / width, "h": (y2 - y1) / height,
            "name": best if known else UNKNOWN, "best": best, "score": round(score, 3), "known": known,
        })
        if log_enabled:
            attendance.record(best if known else UNKNOWN, score)
    return results


app = FastAPI(title="FaceID Lab")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/info")
def info():
    return {
        "people": [str(label) for label in gallery.labels],
        "threshold": gallery.threshold,
        "providers": face_app.models["detection"].session.get_providers()[:1],
        "report": gallery.report,
        "log": attendance.recent[:8],
    }


@app.get("/api/log.csv")
def download_log():
    path = attendance.path()
    if not os.path.exists(path):
        return PlainTextResponse("time,name,score\n", media_type="text/csv")
    return FileResponse(path, media_type="text/csv", filename=os.path.basename(path))


@app.websocket("/ws")
async def stream(websocket: WebSocket):
    await websocket.accept()
    settings = {"threshold": gallery.threshold, "log": True}
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text"):
                update = json.loads(message["text"])
                if "threshold" in update:
                    settings["threshold"] = min(0.95, max(0.05, float(update["threshold"])))
                if "log" in update:
                    settings["log"] = bool(update["log"])
                continue
            if not message.get("bytes"):
                continue
            started = time.perf_counter()
            try:
                frame = decode_image(message["bytes"])
            except ValueError:
                continue
            faces = await asyncio.to_thread(recognize, frame, settings["threshold"], settings["log"])
            await websocket.send_json({
                "faces": faces,
                "ms": round((time.perf_counter() - started) * 1000),
                "log": attendance.recent[:8],
            })
    except WebSocketDisconnect:
        pass


# ---- Đăng ký người mới ----------------------------------------------------------------------

enroll_sessions = {}


class ClipRequest(BaseModel):
    session: str | None = None
    name: str
    frames: list[str]  # ảnh JPEG dạng data URL hoặc base64


class CommitRequest(BaseModel):
    session: str


def analyze_clip(frames):
    stats = {"ok": 0, "no_face": 0, "multiple_faces": 0, "low_score": 0, "small": 0, "blurry": 0, "pose": 0}
    embeddings = []
    for encoded in frames:
        try:
            frame = decode_image(base64.b64decode(encoded.split(",", 1)[-1]))
        except (ValueError, base64.binascii.Error):
            continue
        with face_lock:
            faces = face_app.get(frame)
        if not faces:
            stats["no_face"] += 1
            continue
        if len(faces) != 1:
            stats["multiple_faces"] += 1
            continue
        passes, reason = face_passes_quality(faces[0], frame)
        if not passes:
            stats[reason] += 1
            continue
        embedding = normalized_embedding(faces[0])
        if embedding is not None:
            embeddings.append(embedding)
            stats["ok"] += 1
    return stats, embeddings


@app.post("/api/enroll/clip")
async def enroll_clip(request: ClipRequest):
    name = request.name.strip()
    if not name or len(name) > 40 or any(ch in name for ch in "/\\"):
        raise HTTPException(400, "Tên không hợp lệ (1–40 ký tự, không chứa / hoặc \\).")
    session = enroll_sessions.get(request.session) if request.session else None
    if session is None or session["name"] != name:
        session_id = uuid.uuid4().hex
        session = {"name": name, "clips": [], "id": session_id}
        enroll_sessions[session_id] = session

    stats, embeddings = await asyncio.to_thread(analyze_clip, request.frames)
    clip_index = len(session["clips"]) + 1
    session["clips"].append({"stats": stats, "embeddings": embeddings, "total": len(request.frames)})

    summary = {"session": session["id"], "clip": clip_index, "stats": stats, "total": len(request.frames)}
    all_embeddings = [e for clip in session["clips"] for e in clip["embeddings"]]
    summary["embeddings"] = len(all_embeddings)
    if all_embeddings:
        mean = np.mean(all_embeddings, axis=0)
        mean /= np.linalg.norm(mean)
        others = gallery.labels != name
        if np.any(others):
            similarities = gallery.prototypes[others] @ mean
            best = int(np.argmax(similarities))
            summary["nearest"] = {"name": str(gallery.labels[others][best]), "score": round(float(similarities[best]), 3)}
        summary["exists"] = bool(np.any(gallery.labels == name))
    return summary


@app.post("/api/enroll/commit")
async def enroll_commit(request: CommitRequest):
    session = enroll_sessions.get(request.session)
    if session is None:
        raise HTTPException(404, "Phiên đăng ký đã hết hạn; hãy quay lại video.")
    clips = [clip for clip in session["clips"] if clip["embeddings"]]
    if not clips:
        raise HTTPException(400, "Chưa có frame đạt chất lượng nào.")

    name = session["name"]
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_embeddings, new_labels, new_videos = [], [], []
    for index, clip in enumerate(clips, start=1):
        video_id = f"web/{name}/{stamp}_{index}"
        new_embeddings.extend(clip["embeddings"])
        new_labels.extend([name] * len(clip["embeddings"]))
        new_videos.extend([video_id] * len(clip["embeddings"]))

    data = gallery.embeddings
    if data is None:
        raise HTTPException(500, f"Thiếu {EMBEDDINGS_FILE}; không thể dựng lại prototype.")
    embeddings = np.concatenate([data["embeddings"], np.asarray(new_embeddings, dtype=np.float32)])
    labels = np.concatenate([data["labels"], np.asarray(new_labels, dtype=str)])
    video_ids = np.concatenate([data["video_ids"], np.asarray(new_videos, dtype=str)])

    prototype_labels, prototypes = build_prototypes(embeddings, labels, video_ids)
    np.savez_compressed(
        EMBEDDINGS_FILE, signature=np.asarray(data["signature"]),
        embeddings=embeddings, labels=labels, video_ids=video_ids,
    )
    np.savez_compressed(
        PROTOTYPES_FILE, labels=prototype_labels, prototypes=prototypes,
        cosine_threshold=np.asarray(gallery.threshold, dtype=np.float32),
    )
    await asyncio.to_thread(gallery.reload)
    del enroll_sessions[request.session]
    return {"name": name, "clips": len(clips), "embeddings": len(new_embeddings), "people": len(prototype_labels)}


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

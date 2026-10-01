import ctypes
import glob
import os
import site

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


_DLL_DIRECTORY_HANDLES = []
_CUDNN_DLL_HANDLES = []
_DLL_DIRECTORIES_REGISTERED = False


def register_nvidia_dll_directories():
    global _DLL_DIRECTORIES_REGISTERED
    if os.name != "nt" or _DLL_DIRECTORIES_REGISTERED:
        return

    site_packages_paths = site.getsitepackages()
    for site_packages in site_packages_paths:
        pattern = os.path.join(site_packages, "nvidia", "*", "bin")
        for dll_directory in glob.glob(pattern):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_directory))
            except OSError:
                continue

    for site_packages in site_packages_paths:
        pattern = os.path.join(site_packages, "nvidia", "cudnn", "bin", "*.dll")
        for dll_path in glob.glob(pattern):
            try:
                _CUDNN_DLL_HANDLES.append(ctypes.WinDLL(dll_path))
            except OSError as exc:
                raise RuntimeError(f"Không nạp được thư viện cuDNN: {dll_path}: {exc}") from exc

    _DLL_DIRECTORIES_REGISTERED = True


def create_face_app():
    register_nvidia_dll_directories()
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        try:
            preload_dlls()
        except Exception as exc:
            print(f"[WARN] Không nạp trước được DLL ONNX Runtime: {exc}")

    available_providers = ort.get_available_providers()
    print(f"[INFO] ONNX Runtime providers: {available_providers}")

    providers = ["CPUExecutionProvider"]
    use_cuda = os.environ.get("INSIGHTFACE_USE_CUDA", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if use_cuda and "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError("CUDAExecutionProvider không khả dụng trong ONNX Runtime.")
    if use_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
        print(f"[INFO] Đang dùng provider: {providers}")
        return app
    except Exception as exc:
        if use_cuda:
            raise RuntimeError(f"Không thể khởi tạo InsightFace bằng CUDA: {exc}") from exc
        print(f"[WARN] Không thể khởi tạo InsightFace với providers {providers}: {exc}")
        print("[INFO] Fallback sang CPUExecutionProvider")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app


def normalized_embedding(face):
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        return None
    return embedding / norm
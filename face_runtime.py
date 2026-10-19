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
                print(f"[WARN] Không nạp được thư viện cuDNN: {dll_path}: {exc}")

    _DLL_DIRECTORIES_REGISTERED = True


def _cuda_mode():
    # INSIGHTFACE_USE_CUDA: "1" (mặc định) thử GPU rồi fallback CPU,
    # "0" chỉ dùng CPU, "require" báo lỗi nếu không chạy được GPU.
    value = os.environ.get("INSIGHTFACE_USE_CUDA", "1").strip().lower()
    if value in {"0", "false", "no", "cpu"}:
        return "off"
    if value in {"require", "required", "force"}:
        return "require"
    return "auto"


def _build_app(providers):
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
    print(f"[INFO] Đang dùng provider: {providers}")
    return app


def create_face_app():
    cuda_mode = _cuda_mode()
    if cuda_mode != "off":
        register_nvidia_dll_directories()
        preload_dlls = getattr(ort, "preload_dlls", None)
        if callable(preload_dlls):
            try:
                preload_dlls()
            except Exception as exc:
                print(f"[WARN] Không nạp trước được DLL ONNX Runtime: {exc}")

    available_providers = ort.get_available_providers()
    print(f"[INFO] ONNX Runtime providers: {available_providers}")

    cpu_providers = ["CPUExecutionProvider"]
    if cuda_mode == "off":
        return _build_app(cpu_providers)

    if "CUDAExecutionProvider" not in available_providers:
        message = "CUDAExecutionProvider không khả dụng trong ONNX Runtime."
        if cuda_mode == "require":
            raise RuntimeError(message)
        print(f"[WARN] {message} Fallback sang CPUExecutionProvider (sẽ chậm hơn).")
        return _build_app(cpu_providers)

    try:
        return _build_app(["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception as exc:
        if cuda_mode == "require":
            raise RuntimeError(f"Không thể khởi tạo InsightFace bằng CUDA: {exc}") from exc
        print(f"[WARN] Không thể khởi tạo InsightFace bằng CUDA: {exc}")
        print("[INFO] Fallback sang CPUExecutionProvider (sẽ chậm hơn).")
        return _build_app(cpu_providers)


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
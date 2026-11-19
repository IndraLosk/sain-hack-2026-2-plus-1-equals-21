"""Эмбеддинги BAAI/bge-m3 на ONNX Runtime — только CPU, без torch/transformers.

* токенизация — `tokenizers` (Rust), модель — `onnxruntime` (граф XLMRoberta);
* CLS-pooling + L2-нормализация (как у bge-m3 в sentence-transformers);
* веса скачиваются один раз в `data/models/bge-m3-onnx` и дальше берутся из кэша.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .config import get_settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_EMBEDDER: Optional["OnnxEmbedder"] = None

# файлы, которые нужны для инференса (скачиваются из HF-репозитория модели)
_MODEL_FILES = ("onnx/model.onnx", "onnx/model.onnx_data", "onnx/tokenizer.json", "onnx/config.json")


class OnnxEmbedder:
    def __init__(self, model_dir: Path, *, max_length: int = 512, batch_size: int = 8, threads: int = 0) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.max_length = max_length
        self.batch_size = max(1, batch_size)

        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        model_path = model_dir / "onnx" / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(threads or (os.cpu_count() or 4), 8))
        options.inter_op_num_threads = 1
        logger.info("loading ONNX embedder %s (threads=%s)", model_path, options.intra_op_num_threads)
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
        vectors = hidden[:, 0, :].astype(np.float32)  # CLS-pooling
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        batches = [
            self._encode_batch(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(batches).tolist()

    @property
    def dimension(self) -> int:
        return int(self.session.get_outputs()[0].shape[-1] or 1024)


def _model_dir() -> Path:
    """Локальный каталог с tokenizer.json + onnx/model.onnx (скачивает при необходимости)."""
    settings = get_settings()
    if settings.embedding_onnx_path:
        return Path(settings.embedding_onnx_path)
    target = Path(settings.chroma_dir).parent / "models" / "bge-m3-onnx"
    if (target / "tokenizer.json").exists() and (target / "onnx" / "model.onnx").exists():
        return target
    return _download(target, settings.embedding_model)


def _download(target: Path, repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("downloading %s (ONNX, ~2.3 ГБ) into %s — один раз", repo_id, target)
    for remote in _MODEL_FILES:
        local = hf_hub_download(repo_id=repo_id, filename=remote, cache_dir=str(target / "_hf"))
        dest = target / Path(remote).name
        if dest.exists():
            continue
        try:
            os.link(local, dest)  # дешёвая жёсткая ссылка из HF-кэша
        except OSError:
            shutil.copyfile(local, dest)
    onnx_dir = target / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    for name in ("model.onnx", "model.onnx_data"):
        src = target / name
        if src.exists() and not (onnx_dir / name).exists():
            src.replace(onnx_dir / name)
    return target


def get_embedder() -> OnnxEmbedder:
    """Ленивая загрузка ONNX-модели — один раз на процесс."""
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    with _LOCK:
        if _EMBEDDER is None:
            settings = get_settings()
            _EMBEDDER = OnnxEmbedder(
                _model_dir(),
                max_length=settings.embedding_max_length,
                batch_size=settings.embedding_batch_size,
                threads=settings.embedding_threads,
            )
    return _EMBEDDER


def embed_texts(texts: Sequence[str], is_query: bool = False) -> List[List[float]]:
    """Эмбеддинги документов/запросов (bge-m3 не требует префиксов)."""
    if not texts:
        return []
    return get_embedder().encode(list(texts))


__all__ = ["OnnxEmbedder", "get_embedder", "embed_texts"]

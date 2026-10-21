import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Callable, List
from collections import deque
import struct
from functools import wraps
import inspect


def get_default_args(func):
    signature = inspect.signature(func)
    default_args = {}
    for name, parameter in signature.parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            default_args[name] = parameter.default
    return default_args

def stable_hash(obj: Any) -> str:
    dumped = json.dumps(
        obj,
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

def atomic_write_json(path: Path, data: Dict):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

class HalDetTransformCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._open_sig = None
    
    def _sig_dir(self, sig: str) -> Path:
        return self.cache_dir / sig
    def _meta_path(self, sig: str) -> Path:
        return self._sig_dir(sig) / "meta.json"
    def _offsets_path(self, sig: str) -> Path:
        return self._sig_dir(sig) / "offsets.json"
    def _data_path(self, sig: str) -> Path:
        return self._sig_dir(sig) / "data.bin"

    def _open_signature(self, sig: str, config: Dict):
        if self._open_sig == sig:
            return
        self._flush()
        sig_dir = self._sig_dir(sig)
        sig_dir.mkdir(parents=True, exist_ok=True)

        meta_path = self._meta_path(sig)
        if not meta_path.exists():
            meta = {
                "config_signature": sig,
                "created_at": datetime.now().isoformat(),
                "config": config,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
        else:
            meta = json.loads(meta_path.read_text())
            if meta["config_signature"] != sig:
                raise RuntimeError("Corrupted meta.json")

        offsets_path = self._offsets_path(sig)
        if offsets_path.exists():
            self._offsets = json.loads(offsets_path.read_text())
        else:
            self._offsets = {}

        self._data_fp = open(self._data_path(sig), "ab+")

        self._open_sig = sig
        self._offsets_dirty = False

    def _flush(self):
        if self._open_sig is None:
            return
        if self._offsets_dirty:
            self._offsets_path(self._open_sig).write_text(
                json.dumps(self._offsets)
            )
            self._offsets_dirty = False
        if self._data_fp:
            self._data_fp.flush()

    def __del__(self):
        try:
            self._flush()
            if self._data_fp:
                self._data_fp.close()
        except Exception:
            pass

    def get(self, sig: str, sample_key: str) -> Optional[List[float]]:
        offsets_path = self._offsets_path(sig)
        data_path = self._data_path(sig)
        if not offsets_path.exists() or not data_path.exists():
            return None
        if self._open_sig != sig:
            self._open_signature(sig, config={})
        entry = self._offsets.get(sample_key)
        if entry is None:
            return None
        offset, length = entry
        self._data_fp.seek(offset)
        raw = self._data_fp.read(4 + 4 * length)
        n = struct.unpack("<i", raw[:4])[0]
        values = struct.unpack(f"<{n}f", raw[4:])
        return list(values)

    def set(
        self,
        sig: str,
        config: Dict,
        sample_key: str,
        value: List[float],
    ):
        self._open_signature(sig, config)
        if sample_key in self._offsets:
            return  
        value = [float(f"{v:.6f}") for v in value]
        self._data_fp.seek(0, 2)
        offset = self._data_fp.tell()
        payload = struct.pack("<i", len(value))
        payload += struct.pack(f"<{len(value)}f", *value)
        self._data_fp.write(payload)
        self._offsets[sample_key] = (offset, len(value))
        self._offsets_dirty = True

    def __call__(self, fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) != 1:
                raise RuntimeError(
                    f"All inputs of {fn.__name__} must be keyword-only for correct caching."
                )
            self_obj = args[0]
            stream = kwargs.pop("stream")
            config = self_obj.config.copy()
            config["method"] = self_obj.__class__.__name__
            config["method_type"] = self_obj.method_type
            config_signature = stable_hash(config)
            batch_size = self_obj.batch_size
            next_emit_idx = 0
            ready: Dict[int, Any] = {}
            pending: deque[tuple[int, Any]] = deque()
            emit_buffer: List[Any] = []
            def filtered_stream():
                for idx, sample in enumerate(stream):
                    sample_key = stable_hash(sample)
                    cached = self.get(config_signature, sample_key)
                    if cached is not None:
                        ready[idx] = cached
                    else:
                        pending.append((idx, sample))
                        yield sample
            fn_outputs_stream = fn(self_obj, stream=filtered_stream(), **kwargs)
            for fn_outputs_batch in fn_outputs_stream:
                for output in fn_outputs_batch:
                    idx, sample = pending.popleft()
                    sample_key = stable_hash(sample)
                    self.set(config_signature, config, sample_key, output)
                    ready[idx] = output
                while next_emit_idx in ready:
                    emit_buffer.append(ready.pop(next_emit_idx))
                    next_emit_idx += 1
                    if len(emit_buffer) == batch_size:
                        yield emit_buffer
                        emit_buffer = []
                self._flush()
            while next_emit_idx in ready:
                emit_buffer.append(ready.pop(next_emit_idx))
                next_emit_idx += 1
                if len(emit_buffer) == batch_size:
                    yield emit_buffer
                    emit_buffer = []
            if emit_buffer:
                yield emit_buffer
            self._flush()
        return wrapped

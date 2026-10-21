import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Callable
from functools import wraps
import inspect
from typing import cast
from .llm_utils import check_prompt_response_compatibility


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


class LLMResponseCache:
    """
    cache_dir/
      <config_signature>/
        meta.json
        shard_000.json
        shard_001.json
    """

    def __init__(self, cache_dir: str, max_shard_size_bytes: int = 100 * 1024 * 8):
        self.cache_dir = Path(cache_dir) / "responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_shard_size_bytes

        self._open_meta_sig = None
        self._open_meta = None
        self._open_meta_dirty = False

        self._open_index_sig = None
        self._open_index = None
        self._open_index_dirty = False

        self._open_shard_sig = None
        self._open_shard_name = None
        self._open_shard_data = None
        self._open_shard_dirty = False

    def _meta_path(self, sig: str) -> Path:
        return self.cache_dir / sig / "meta.json"
    
    def _index_path(self, sig: str) -> Path:
        return self.cache_dir / sig / "index.json"
    
    def _shard_path(self, sig: str, name: str) -> Path:
        return self.cache_dir / sig / name
    
    def _flush_meta(self):
        if (
            self._open_meta_sig is None or
            self._open_meta is None or
            self._open_meta_dirty is False
        ):
            return
        atomic_write_json(self._meta_path(self._open_meta_sig), self._open_meta)
        self._open_meta_dirty = False

    def _flush_index(self):
        if (
            self._open_index_sig is None or
            self._open_index is None or
            self._open_index_dirty is False
        ):
            return
        atomic_write_json(self._index_path(self._open_index_sig), self._open_index)
        self._open_index_dirty = False

    def _flush_shard(self):
        if (
            self._open_shard_name is None or
            self._open_shard_sig is None or
            self._open_shard_data is None or
            self._open_shard_dirty is False
        ):
            return
        atomic_write_json(
            self._shard_path(self._open_shard_sig, self._open_shard_name),
            self._open_shard_data,
        )
        self._open_shard_dirty = False

    def __del__(self):
        try:
            self._flush_shard()
            self._flush_meta()
            self._flush_index()
        except Exception:
            pass

    def _create_meta(self, model: str, sig: str, config: Dict) -> Dict:
        sig_dir = self.cache_dir / sig
        sig_dir.mkdir(parents=True, exist_ok=True)
        self._flush_meta()
        meta = {
            "model": model,
            "config_signature": sig,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "max_shard_size_bytes": self.max_size,
            "shards": [],
        }
        atomic_write_json(self._meta_path(sig), meta)
        self._open_meta_sig = sig
        self._open_meta = meta
        self._open_meta_dirty = False
        return meta
    
    def _create_index(self, sig: str) -> Dict:
        sig_dir = self.cache_dir / sig
        sig_dir.mkdir(parents=True, exist_ok=True)
        self._flush_index()
        index = dict()
        atomic_write_json(self._index_path(sig), index)
        self._open_index_sig = sig
        self._open_index = index
        self._open_index_dirty = False
        return index
    
    def _create_shard(self, sig: str, meta: Dict) -> Dict:
        self._flush_shard()

        idx = len(meta["shards"])
        name = f"shard_{idx:03d}.json"
        shard = dict()

        atomic_write_json(self._shard_path(sig, name), shard)

        meta["shards"].append({"name": name, "size": 0})
        self._open_meta_dirty = True

        self._open_shard_sig = sig
        self._open_shard_name = name
        self._open_shard_data = shard
        self._open_shard_dirty = False
        return shard
    
    def _load_meta(self, model: str, sig: str) -> Dict:
        if self._open_meta_sig == sig:
            self._open_meta = cast(dict, self._open_meta)
            return self._open_meta
        self._flush_meta()
        path = self._meta_path(sig)
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        config = meta["config"].copy()
        config["model"] = model
        if stable_hash(config) != meta["config_signature"]:
            raise RuntimeError(f"Corrupted meta.json for signature {sig}")
        self._open_meta_sig = sig
        self._open_meta = meta
        self._open_meta_dirty = False
        return meta
    
    def _load_index(self, sig: str) -> Dict:
        if self._open_index_sig == sig:
            self._open_index = cast(dict, self._open_index)
            return self._open_index
        self._flush_index()
        path = self._index_path(sig)
        with path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        self._open_index_sig = sig
        self._open_index = index
        self._open_index_dirty = False
        return index
    
    def _load_shard(self, sig: str, name: str) -> Dict:
        if self._open_shard_sig == sig and self._open_shard_name == name:
            self._open_shard_data = cast(dict, self._open_shard_data)
            return self._open_shard_data
        self._flush_shard()
        path = self._shard_path(sig, name)
        with path.open("r", encoding="utf-8") as f:
            shard = json.load(f)

        self._open_shard_sig = sig
        self._open_shard_name = name
        self._open_shard_data = shard
        self._open_shard_dirty = False
        return shard

    def get(self, sig: str, sample_key: str) -> Optional[Dict]:
        index_path = self._index_path(sig)
        if not index_path.exists():
            return None
        index = self._load_index(sig)
        if sample_key in index:
            shard = self._load_shard(sig, index[sample_key])
            return shard[sample_key]
        return None
    
    def flush(self):
        self._flush_index()
        self._open_index_dirty = False
        self._flush_meta()
        self._open_meta_dirty = False
        self._flush_shard()
        self._open_shard_dirty = False

    def set(self, model: str, sig: str, config: Dict, sample_key: str, value: Dict):
        meta_path = self._meta_path(sig)
        if not meta_path.exists():
            meta = self._create_meta(model, sig, config)
        else:
            meta = self._load_meta(model, sig)
        if not meta["shards"]:
            shard = self._create_shard(sig, meta)
            shard_info = meta["shards"][-1]
        else:
            shard_info = meta["shards"][-1]
            shard = self._load_shard(sig, shard_info["name"])
        entry_size = len(
            json.dumps({sample_key: value}, ensure_ascii=False).encode("utf-8")
        )
        if shard_info["size"] + entry_size > self.max_size:
            shard = self._create_shard(sig, meta)
            shard_info = meta["shards"][-1]

        index_path = self._index_path(sig)
        if not index_path.exists():
            index = self._create_index(sig)
        else:
            index = self._load_index(sig)
        index[sample_key] = shard_info["name"]
        self._open_index_dirty = True
        shard[sample_key] = value
        self._open_shard_dirty = True
        shard_info["size"] += entry_size
        self._open_meta_dirty = True

    def __call__(self, fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if len(args) != 1:
                raise RuntimeError(
                    f"All inputs of {fn.__name__} must be keyword-only for correct caching."
                )
            fn_kwargs = get_default_args(fn)
            fn_kwargs.update(kwargs)
            self_obj = args[0]
            user_prompt = fn_kwargs.pop("user_prompt")
            assistant_response = fn_kwargs.pop("assistant_response", None)
            if assistant_response is not None and all(map(lambda x: x is None, assistant_response)):
                assistant_response = None
            check_prompt_response_compatibility(user_prompt, assistant_response)
            config = {
                k: v for k, v in fn_kwargs.items()
                if k not in ("batch_size", ...)
            }
            config["model"] = repr(self_obj)
            config_signature = stable_hash(config)
            config.pop("model")
            batch_size = fn_kwargs.get("batch_size", 1)
            idx = 0
            n = len(user_prompt)
            while idx < n:
                user_prompt_sample = user_prompt[idx]
                assistant_response_sample = (
                    assistant_response[idx] if assistant_response is not None else None
                )
                input = {
                    "prompt_history": user_prompt_sample,
                    "response_history": assistant_response_sample,
                }
                sample_key = stable_hash(input)
                cache = self.get(config_signature, sample_key)
                if cache is not None:
                    yield cache
                    idx += 1
                    continue
                batch_user_prompt = []
                batch_assistant_response = [] if assistant_response is not None else None
                while idx < n and len(batch_user_prompt) < batch_size:
                    user_prompt_sample = user_prompt[idx]
                    assistant_response_sample = (
                        assistant_response[idx] if assistant_response is not None else None
                    )
                    input = {
                        "prompt_history": user_prompt_sample,
                        "response_history": assistant_response_sample,
                    }
                    sample_key = stable_hash(input)
                    cache = self.get(config_signature, sample_key)
                    if cache is not None:
                        break
                    batch_user_prompt.append(user_prompt_sample)
                    if batch_assistant_response is not None:
                        batch_assistant_response.append(assistant_response_sample)
                    idx += 1
                for i, (user_prompt_sample, output) in enumerate(zip(
                    batch_user_prompt,
                    fn(self_obj, batch_user_prompt, batch_assistant_response, **fn_kwargs)
                )):
                    assistant_response_sample = batch_assistant_response[i] if batch_assistant_response is not None else None
                    input = {
                        "prompt_history": user_prompt_sample,
                        "response_history": assistant_response_sample,
                    }
                    sample_key = stable_hash(input)
                    output.update(input)
                    self.set(repr(self_obj), config_signature, config, sample_key, output)
                    yield output
                self.flush()
        return wrapped

"""
BOSHRA AI V4.0 - Sovereign Offline Intelligence Matrix
======================================================
Single-file Android assistant (Python 3 / Kivy / Buildozer / arm64-v8a).

Architecture (logical modules inside this file):
  core        - paths, config, crash guard, SQLite store, encryption
  memory      - layered local memory + hashed semantic vectors
  knowledge   - offline document vault (PDF/DOCX/TXT/CSV/images)
  llm         - model discovery, llama.cpp engine, heuristic fallback
  voice       - Vosk recognition, wake word, VAD, language routing
  tts         - Android TTS / Piper / desktop fallback speech queue
  security    - permissions, risk classes, consent, audit, shadow mode
  skills      - manifest validation, registry, sandbox policy, built-ins
  automation  - workflows, routines, shadow-mode previews
  android     - native bridge through Pyjnius (device, battery, intents)
  ui          - neon/glass Kivy shell, GLSL aurora + living sphere

Design rules enforced here:
  * No cloud dependency for core operation; online tools fail closed.
  * No LLM-generated code is executed. Skills are manifests or isolated
    processes that the user explicitly authorises.
  * Heavy native models load lazily and unload under memory pressure.
  * All risky/destructive actions have a dry-run preview and confirmation.
  * Optional native components (llama.cpp, Vosk, Piper) are detected, never
    assumed. Missing components produce explicit status, not fake success.

Run locally:      python3 main.py
Self-test:        python3 main.py --self-test
Headless smoke:   BOSHRA_SMOKE=1 xvfb-run -a python3 main.py
Android package:  buildozer android debug   (see README.md)
"""

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import hmac
import html
import io
import json
import math
import os
import queue
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
import wave
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# Kivy global configuration must happen before any Kivy core import.
# --------------------------------------------------------------------------
os.environ.setdefault("KIVY_NO_ARGS", "1")
os.environ.setdefault("KIVY_NO_CONSOLELOG", "0")
os.environ.setdefault("KIVY_LOG_MODE", "PYTHON")

from kivy.config import Config as KivyConfig  # noqa: E402

KivyConfig.set("graphics", "multisamples", "0")
KivyConfig.set("graphics", "resizable", "1")
KivyConfig.set("input", "mouse", "mouse,multitouch_on_demand")
KivyConfig.set("kivy", "exit_on_escape", "0")
KivyConfig.set("widgets", "scroll_timeout", "60")

APP_NAME = "Boshra AI"
APP_VERSION = "4.0.0"
BUILD_STAMP = "2026-11-03"

IS_ANDROID = bool(os.environ.get("ANDROID_ARGUMENT")) or sys.platform == "android"
IS_DESKTOP = not IS_ANDROID

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value) if value else None


def _resolve_private_dir() -> Path:
    if IS_ANDROID:
        base = (
            os.environ.get("ANDROID_PRIVATE")
            or os.environ.get("ANDROID_APP_PATH")
            or "/data/data/org.boshra.boshra/files"
        )
        return Path(base)
    override = os.environ.get("BOSHRA_HOME")
    if override:
        return Path(override)
    return Path.home() / ".boshra_ai"


def _resolve_public_dir() -> Path:
    if IS_ANDROID:
        shared = Path("/storage/emulated/0")
        if shared.exists():
            return shared / "BoshraAI"
        return _resolve_private_dir() / "public"
    return Path.home() / "BoshraAI"


PRIVATE_DIR = _resolve_private_dir()
PUBLIC_DIR = _resolve_public_dir()
MODELS_DIR = PRIVATE_DIR / "models"
SKILLS_DIR = PRIVATE_DIR / "skills"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
ERROR_LOG_PATH = (
    Path("/storage/emulated/0/app_error_log.txt")
    if IS_ANDROID and Path("/storage/emulated/0").exists()
    else PRIVATE_DIR / "app_error_log.txt"
)
PRIVATE_LOG_PATH = PRIVATE_DIR / "boshra.log"
DB_PATH = PRIVATE_DIR / "boshra.db"
CONFIG_PATH = PRIVATE_DIR / "config.json"
EXPORT_DIR = PUBLIC_DIR / "exports"
WORKSPACE_DIR = PUBLIC_DIR / "workspace"
DOWNLOADS_DIR = PUBLIC_DIR / "downloads"
ASSET_NAMES = ("phi3", "vosk-model-ar", "vosk-model-en", "piper-tts")
ASSETS_PUBLIC_DIR = (
    Path("/sdcard/assets") if IS_ANDROID and Path("/sdcard").exists() else ASSETS_DIR
)


def _kivy_user_data_dir() -> Optional[Path]:
    """Kivy's per-app data directory, resolved lazily.

    ``App.user_data_dir`` only exists after the App instance is built, so this
    is queried at call time instead of being cached at import time. On Android
    it points into the app's private storage, which is where Buildozer/p4a
    extract bundled files and where downloads are kept.
    """
    if "kivy.app" not in sys.modules:
        return None
    try:
        from kivy.app import App

        running = App.get_running_app()
        if running is not None:
            value = getattr(running, "user_data_dir", "")
            if value:
                return Path(value).expanduser()
    except Exception:  # noqa: BLE001 - App may be shutting down
        pass
    return None


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    """De-duplicate paths without requiring them to exist."""
    unique: List[Path] = []
    seen: set = set()
    for raw in paths:
        try:
            path = Path(raw).expanduser()
            key = os.path.normcase(str(path.resolve(strict=False)))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _saved_assets_root() -> Optional[Path]:
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(saved, dict):
        return None
    value = saved.get("assets_path")
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    return None


def asset_roots() -> List[Path]:
    """Ordered asset roots for desktop and Android; never raises.

    Desktop: the ``assets`` folder next to this script is preferred, so a
    source checkout is self-contained. Kivy's user data directory and the
    private data directory are searched as fallbacks.

    Android (Buildozer/p4a): ``__file__`` lives in app-private storage where
    bundled files are extracted, so the script-relative ``assets`` folder is
    still valid. A user-visible ``/sdcard/assets`` folder is preferred when it
    exists so models can be copied over USB without root; the Kivy user data
    directory and ``ANDROID_PRIVATE`` cover installs that keep assets internal.

    A saved ``assets_path`` from Settings and the ``BOSHRA_ASSETS`` environment
    override are honoured first. Roots that do not exist yet are still
    returned as safe creation targets, and filesystem errors are swallowed so
    startup cannot fail with ``FileNotFoundError``.
    """
    candidates: List[Path] = []
    override = _env_path("BOSHRA_ASSETS") or _env_path("BOSHRA_ASSETS_PATH")
    if override:
        candidates.append(override)

    saved = _saved_assets_root()
    if saved is not None:
        candidates.append(saved)

    user_data = _kivy_user_data_dir()
    if IS_ANDROID:
        # External storage, then packaged/app-private locations.
        shared_assets = Path("/sdcard/assets")
        if shared_assets.exists():
            candidates.append(shared_assets)
        android_public = _env_path("ANDROID_PUBLIC")
        if android_public:
            candidates.append(android_public / "assets")
        candidates.append(ASSETS_DIR)
        candidates.append(PRIVATE_DIR / "assets")
        candidates.append(Path("/storage/emulated/0/BoshraAI/assets"))
    else:
        candidates.append(ASSETS_DIR)
        try:
            candidates.append(Path.cwd() / "assets")
        except OSError:
            pass
        candidates.append(PUBLIC_DIR / "assets")
        candidates.append(PRIVATE_DIR / "assets")
    if user_data is not None:
        candidates.append(user_data / "assets")

    roots = _dedupe_paths(candidates)
    return roots or [ASSETS_DIR]


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _asset_creation_roots() -> List[Path]:
    """Only the roots where Boshra should actually create model folders.

    ``asset_roots`` also lists read-only fallbacks for discovery; creating
    folders in all of them would litter the home directory, so creation is
    limited to an explicit user choice and the platform's normal storage.
    """
    candidates: List[Path] = []
    override = _env_path("BOSHRA_ASSETS") or _env_path("BOSHRA_ASSETS_PATH")
    if override:
        candidates.append(override)
    saved = _saved_assets_root()
    if saved is not None:
        candidates.append(saved)
    if IS_ANDROID:
        shared = Path("/sdcard/assets")
        try:
            if Path("/sdcard").is_dir():
                candidates.append(shared)
        except OSError:
            pass
        candidates.append(ASSETS_DIR)
        candidates.append(PRIVATE_DIR / "assets")
    else:
        candidates.append(ASSETS_DIR)
    return _dedupe_paths(candidates) or [ASSETS_DIR]


def ensure_asset_dirs() -> Dict[str, Path]:
    """Create the model folders the app scans, preferring shared storage on Android."""
    created: Dict[str, Path] = {}
    for root in _asset_creation_roots():
        for name in ASSET_NAMES:
            folder = root / name
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            created.setdefault(name, folder)
    return created


def ensure_dirs() -> None:
    for path in (
        PRIVATE_DIR,
        PUBLIC_DIR,
        MODELS_DIR,
        SKILLS_DIR,
        EXPORT_DIR,
        WORKSPACE_DIR,
        DOWNLOADS_DIR,
        PRIVATE_DIR / "logs",
        PRIVATE_DIR / "cache",
        PRIVATE_DIR / "backups",
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


ensure_dirs()
ensure_asset_dirs()

# --------------------------------------------------------------------------
# Logging + crash guard
# --------------------------------------------------------------------------
class BoshraLogger:
    """Minimal, dependency-free logger that never raises."""

    _lock = threading.RLock()
    _LEVELS = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}

    def __init__(self, path: Path = PRIVATE_LOG_PATH, level: str = "INFO") -> None:
        self.path = path
        self.level = self._LEVELS.get(level, 20)

    def _write(self, level: str, message: str) -> None:
        if self._LEVELS.get(level, 20) < self.level:
            return
        line = f"{datetime.now().isoformat(timespec='seconds')} {level:<5} {message}\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists() and self.path.stat().st_size > 2_000_000:
                    rotated = self.path.with_suffix(".old.log")
                    with contextlib.suppress(OSError):
                        os.replace(self.path, rotated)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError:
                pass

    def debug(self, message: str) -> None:
        self._write("DEBUG", message)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARN", message)

    def error(self, message: str, exc: Optional[BaseException] = None) -> None:
        self._write("ERROR", message)
        if exc is not None:
            self._write("ERROR", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


LOG = BoshraLogger()


def _append_error_dump(title: str, details: str) -> None:
    stamp = datetime.now().isoformat(timespec="seconds")
    block = f"\n===== {stamp} | {title} | v{APP_VERSION} =====\n{details}\n"
    for path in (PRIVATE_LOG_PATH, ERROR_LOG_PATH):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(block)
        except OSError:
            pass


def install_crash_guard() -> None:
    original_hook = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _append_error_dump("uncaught-exception", details)
        try:
            original_hook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = hook

    def thread_hook(args):
        details = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        _append_error_dump(f"thread-exception:{args.thread.name if args.thread else '?'}", details)

    threading.excepthook = thread_hook


install_crash_guard()

# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_iso(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def human_ts(value: str) -> str:
    dt = parse_iso(value)
    if not dt:
        return value
    return dt.strftime("%Y-%m-%d %H:%M")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_json_load(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        LOG.error(f"atomic_write failed for {path}", exc)
        with contextlib.suppress(OSError):
            if "tmp" in locals() and os.path.exists(tmp):
                os.unlink(tmp)
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def human_bytes(size: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


# --------------------------------------------------------------------------
# Asset inventory: cross-platform discovery and validation
# --------------------------------------------------------------------------
_ASSET_EXTRA_FOLDERS: Dict[str, Tuple[Path, ...]] = {
    # Locations older builds and manual installs may still use.
    "phi3": (
        MODELS_DIR, PUBLIC_DIR / "models", Path("/sdcard/phi3"),
        Path("/sdcard/BoshraAI/models"),
    ),
    "vosk-model-ar": (
        PRIVATE_DIR / "vosk-model-ar", Path("/sdcard/BoshraAI/models/vosk-model-ar"),
    ),
    "vosk-model-en": (
        PRIVATE_DIR / "vosk-model-en", Path("/sdcard/BoshraAI/models/vosk-model-en"),
    ),
    "piper-tts": (PRIVATE_DIR / "piper-tts", Path("/sdcard/assets/piper-tts")),
    "fonts": (FONTS_DIR,),
}

_VOSK_MARKERS = ("am", "conf", "graph")
_VOSK_KEY_FILES = (
    ("am/final.mdl", ("am", "final.mdl")),
    ("conf/model.conf", ("conf", "model.conf")),
    ("conf/mfcc.conf", ("conf", "mfcc.conf")),
    ("graph/HCLG.fst", ("graph", "HCLG.fst")),
    ("graph/words.txt", ("graph", "words.txt")),
)


@dataclass
class AssetStatus:
    """One expected asset folder and what validation found there."""

    key: str
    label: str
    path: Path
    ok: bool = False
    found: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "path": str(self.path),
            "ok": self.ok,
            "found": list(self.found),
            "missing": list(self.missing),
            "detail": self.detail,
        }


@dataclass
class AssetReport:
    """Complete validation report; safe to serialize or show in the UI."""

    root: Path
    roots: List[Path] = field(default_factory=list)
    statuses: List[AssetStatus] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.statuses) and all(item.ok for item in self.statuses)

    @property
    def missing(self) -> List[str]:
        return [item.key for item in self.statuses if not item.ok]

    def status(self, key: str) -> Optional[AssetStatus]:
        return next((item for item in self.statuses if item.key == key), None)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "root": str(self.root),
            "roots": [str(path) for path in self.roots],
            "missing": self.missing,
            "error": self.error,
            "assets": [item.as_dict() for item in self.statuses],
        }


def _asset_folders(
    name: str, roots: Sequence[Path], include_extras: bool = True,
) -> List[Path]:
    candidates = [root / name for root in roots]
    if include_extras:
        candidates.extend(_ASSET_EXTRA_FOLDERS.get(name, ()))
    return [folder for folder in _dedupe_paths(candidates) if _is_dir(folder)]


def _scan_files(folder: Path, pattern: str, recursive: bool = False) -> List[Path]:
    try:
        iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
        found = [path for path in iterator if _is_file(path)]
    except (OSError, ValueError):
        return []
    return sorted(found, key=lambda path: str(path).lower())


def _gguf_is_valid(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _check_phi3(
    key: str, label: str, roots: Sequence[Path], include_extras: bool = True,
) -> AssetStatus:
    folders = _asset_folders(key, roots, include_extras)
    invalid: List[Path] = []
    for folder in folders:
        models = _dedupe_paths(
            _scan_files(folder, "*.gguf") + _scan_files(folder, "*/*.gguf")
        )
        valid: List[Tuple[Path, int]] = []
        for model in models:
            if not _gguf_is_valid(model):
                invalid.append(model)
                continue
            try:
                size = model.stat().st_size
            except OSError:
                size = 0
            valid.append((model, size))
        if valid:
            model, size = max(valid, key=lambda item: item[1])
            return AssetStatus(
                key, label, model.parent, True, [model.name], [],
                f"{model.name} · {human_bytes(size)}",
            )
    if invalid:
        return AssetStatus(
            key, label, invalid[0].parent, False,
            [model.name for model in invalid[:4]], ["valid GGUF header"],
            f"invalid GGUF header: {invalid[0].name}",
        )
    if folders:
        return AssetStatus(
            key, label, folders[0], False, [], ["*.gguf"],
            "folder found; no .gguf model",
        )
    return AssetStatus(
        key, label, roots[0] / key, False, [], [f"{key} folder"], "folder not found",
    )


def _vosk_layout(folder: Path) -> Optional[Tuple[Path, List[str]]]:
    candidates = [folder]
    try:
        candidates.extend(
            sorted(
                (child for child in folder.iterdir() if _is_dir(child)),
                key=lambda child: child.name.lower(),
            )[:8]
        )
    except OSError:
        pass
    for candidate in candidates:
        dirs = {name: _is_dir(candidate / name) for name in _VOSK_MARKERS}
        files = {
            name: _is_file(candidate.joinpath(*parts))
            for name, parts in _VOSK_KEY_FILES
        }
        if sum(dirs.values()) >= 2 or sum(files.values()) >= 2:
            found = [f"{name}/" for name, present in dirs.items() if present]
            found.extend(name for name, present in files.items() if present)
            return candidate, found
    return None


def _check_vosk(
    key: str, label: str, roots: Sequence[Path], include_extras: bool = True,
) -> AssetStatus:
    folders = _asset_folders(key, roots, include_extras)
    for folder in folders:
        layout = _vosk_layout(folder)
        if layout is not None:
            path, found = layout
            return AssetStatus(key, label, path, True, found, [], ", ".join(found[:4]))
    if folders:
        folder = folders[0]
        found = [f"{name}/" for name in _VOSK_MARKERS if _is_dir(folder / name)]
        missing = [f"{name}/" for name in _VOSK_MARKERS if not _is_dir(folder / name)]
        return AssetStatus(
            key, label, folder, False, found, missing,
            "incomplete Vosk model structure",
        )
    return AssetStatus(
        key, label, roots[0] / key, False, [], [f"{key} folder"], "folder not found",
    )


def _piper_sidecar(voice: Path) -> Optional[Path]:
    for config in (Path(f"{voice}.json"), voice.with_suffix(".json")):
        if _is_file(config):
            return config
    return None


def _check_piper(
    key: str, label: str, roots: Sequence[Path], include_extras: bool = True,
) -> AssetStatus:
    folders = _asset_folders(key, roots, include_extras)
    incomplete: Optional[Tuple[Path, List[Path]]] = None
    for folder in folders:
        voices = _scan_files(folder, "*.onnx", recursive=True)
        if not voices:
            continue
        paired: List[Path] = []
        unpaired: List[Path] = []
        for voice in voices:
            if _piper_sidecar(voice) is not None:
                paired.append(voice)
            else:
                unpaired.append(voice)
        if paired:
            path = paired[0].parent
            detail = f"{len(paired)} voice(s) with JSON config"
            if unpaired:
                detail += f"; {len(unpaired)} without config"
            return AssetStatus(
                key, label, path, True, [voice.name for voice in paired],
                [f"{voice.name}.json" for voice in unpaired], detail,
            )
        if incomplete is None:
            incomplete = (voices[0].parent, voices)
    if incomplete is not None:
        path, voices = incomplete
        return AssetStatus(
            key, label, path, False, [voice.name for voice in voices],
            [f"{voice.name}.json" for voice in voices], "missing Piper voice config",
        )
    if folders:
        return AssetStatus(
            key, label, folders[0], False, [], ["*.onnx"],
            "folder found; no .onnx voice",
        )
    return AssetStatus(
        key, label, roots[0] / key, False, [], [f"{key} folder"], "folder not found",
    )


def _check_fonts(
    key: str, label: str, roots: Sequence[Path], include_extras: bool = True,
) -> AssetStatus:
    folders = _asset_folders(key, roots, include_extras)
    for folder in folders:
        fonts = _scan_files(folder, "*.ttf") + _scan_files(folder, "*.ttc")
        if fonts:
            names = [font.name for font in fonts]
            return AssetStatus(
                key, label, folder, True, names, [],
                f"{len(names)} font file(s)",
            )
    if folders:
        return AssetStatus(
            key, label, folders[0], False, [], ["*.ttf"],
            "folder found; no TrueType font",
        )
    return AssetStatus(
        key, label, roots[0] / key, False, [], ["fonts folder"], "folder not found",
    )


_ASSET_SPECS = (
    ("phi3", "Phi-3 GGUF", _check_phi3),
    ("vosk-model-ar", "Vosk Arabic", _check_vosk),
    ("vosk-model-en", "Vosk English", _check_vosk),
    ("piper-tts", "Piper TTS", _check_piper),
    ("fonts", "Fonts", _check_fonts),
)


def inspect_assets(root: Optional[Path] = None) -> AssetReport:
    """Validate every expected asset folder without ever raising.

    Pass ``root`` to audit a specific assets directory (used by tests and the
    Models screen); otherwise every platform candidate from ``asset_roots()``
    is searched so a working folder cannot be masked by a stale config path.
    """
    try:
        roots = _dedupe_paths([root]) if root is not None else asset_roots()
    except (OSError, TypeError, ValueError):
        roots = [ASSETS_DIR]
    roots = roots or [ASSETS_DIR]
    primary = next((path for path in roots if _is_dir(path)), roots[0])
    statuses: List[AssetStatus] = []
    for key, label, checker in _ASSET_SPECS:
        try:
            status = checker(key, label, roots, root is None)
        except Exception as exc:  # noqa: BLE001 - validation must never crash
            status = AssetStatus(
                key, label, primary / key, False, [], ["check error"],
                f"check failed: {type(exc).__name__}: {exc}",
            )
        statuses.append(status)
    return AssetReport(root=primary, roots=roots, statuses=statuses)


# --------------------------------------------------------------------------
# Text: language detection, Arabic shaping/bidi, i18n
# --------------------------------------------------------------------------
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

try:
    import arabic_reshaper as _arabic_reshaper  # type: ignore

    try:
        from bidi.algorithm import get_display as _bidi_get_display  # type: ignore
    except ImportError:
        _bidi_get_display = None  # type: ignore
except ImportError:
    _arabic_reshaper = None  # type: ignore
    _bidi_get_display = None  # type: ignore


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text or ""))


def text_is_rtl(text: str) -> bool:
    """True when the first strong letter belongs to an Arabic script."""
    for char in text or "":
        if _ARABIC_RE.match(char):
            return True
        if char.isalpha():
            return False
    return False


def text_is_rtl_dominant(text: str) -> bool:
    letters = [char for char in (text or "") if char.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for char in letters if _ARABIC_RE.match(char))
    return arabic / len(letters) >= 0.6


def detect_language(text: str) -> str:
    """Returns 'ar' or 'en' using script ratio; digits-only defaults to en."""
    text = text or ""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for c in letters if _ARABIC_RE.match(c))
    return "ar" if arabic / len(letters) >= 0.35 else "en"


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS_RE.sub("", text or "").replace("\u0640", "")
    return (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ة", "ه")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )


ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def localize_digits(text: str, enabled: bool = True) -> str:
    return str(text).translate(ARABIC_DIGITS) if enabled else str(text)


def hijri_date(moment: Optional[datetime] = None) -> str:
    """Tabular Islamic (Kuwaiti algorithm) date; accurate to about a day."""
    moment = moment or datetime.now()
    year, month, day = moment.year, moment.month, moment.day
    if month < 3:
        year -= 1
        month += 12
    jd = (
        (1461 * (year + 4800)) // 4
        + (367 * (month - 2)) // 12
        - (3 * ((year + 4900) // 100)) // 4
        + day
        - 32075
    )
    l = jd - 1948440 + 10632
    n = (l - 1) // 10631
    l = l - 10631 * n + 354
    j = ((10985 - l) // 5316) * ((50 * l) // 17719) + (l // 5670) * ((43 * l) // 15238)
    l = l - ((30 - j) // 15) * ((17719 * j) // 50) - (j // 16) * ((15238 * j) // 43) + 29
    h_month = (24 * l) // 709
    h_day = l - (709 * h_month) // 24
    h_year = 30 * n + j - 30
    names_ar = (
        "محرم", "صفر", "ربيع الأول", "ربيع الآخر", "جمادى الأولى", "جمادى الآخرة",
        "رجب", "شعبان", "رمضان", "شوال", "ذو القعدة", "ذو الحجة",
    )
    names_en = (
        "Muharram", "Safar", "Rabi I", "Rabi II", "Jumada I", "Jumada II",
        "Rajab", "Sha'ban", "Ramadan", "Shawwal", "Dhu al-Qi'dah", "Dhu al-Hijjah",
    )
    month_index = max(0, min(11, h_month - 1))
    if _UI_LANG == "ar":
        return f"{h_day} {names_ar[month_index]} {h_year} هـ"
    return f"{h_day} {names_en[month_index]} {h_year} AH"


def parse_hhmm(value: str, default: Tuple[int, int] = (22, 0)) -> Tuple[int, int]:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", (value or "").strip())
    if not match:
        return default
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return default
    return hour, minute


def in_quiet_hours(moment: datetime, start: str, end: str) -> bool:
    start_h, start_m = parse_hhmm(start, (22, 0))
    end_h, end_m = parse_hhmm(end, (7, 0))
    now_min = moment.hour * 60 + moment.minute
    start_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m
    if start_min == end_min:
        return False
    if start_min < end_min:
        return start_min <= now_min < end_min
    return now_min >= start_min or now_min < end_min


def split_sentences(text: str, limit: int = 8) -> List[str]:
    parts = re.split(r"(?<=[.!?؟])\s+|\n+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()][:limit]


def approximate_tokens(text: str) -> int:
    return max(1, int(len(text or "") / 4)) if (text or "").strip() else 0


@lru_cache(maxsize=4096)
def shape_text(text: str) -> str:
    """Reshapes Arabic and applies bidi for Kivy's non-Pango text renderer."""
    if not text:
        return ""
    if not contains_arabic(text):
        return text
    if _arabic_reshaper is None:
        return text
    try:
        if "\n" in text:
            return "\n".join(shape_text(line) for line in text.split("\n"))
        reshaped = _arabic_reshaper.reshape(text)
        if _bidi_get_display is not None:
            return _bidi_get_display(reshaped)
        return reshaped
    except Exception:
        return text


I18N: Dict[str, Tuple[str, str]] = {
    "app_tagline": ("Sovereign Offline Intelligence", "ذكاء سيادي دون اتصال"),
    "chat": ("Chat", "المحادثة"),
    "skills": ("Skills", "المهارات"),
    "memory": ("Memory", "الذاكرة"),
    "vault": ("Vault", "الخزنة"),
    "flow": ("Automation", "الأتمتة"),
    "privacy": ("Privacy", "الخصوصية"),
    "diagnostics": ("Diagnostics", "التشخيص"),
    "models": ("Models", "النماذج"),
    "settings": ("Settings", "الإعدادات"),
    "sos": ("Emergency", "الطوارئ"),
    "home": ("Home", "الرئيسية"),
    "send": ("Send", "إرسال"),
    "stop": ("Stop", "إيقاف"),
    "mic": ("Push to talk", "اضغط للتحدث"),
    "listening": ("Listening", "أستمع"),
    "thinking": ("Thinking", "أفكر"),
    "speaking": ("Speaking", "أتحدث"),
    "idle": ("Ready", "جاهز"),
    "offline": ("Offline", "دون اتصال"),
    "online": ("Online", "متصل"),
    "local_only": ("Local only", "محلي فقط"),
    "ask_placeholder": ("Ask Boshra...", "اسأل بشرى..."),
    "remember": ("Remember", "تذكّر"),
    "search": ("Search", "بحث"),
    "import": ("Import", "استيراد"),
    "export": ("Export", "تصدير"),
    "delete": ("Delete", "حذف"),
    "confirm": ("Confirm", "تأكيد"),
    "cancel": ("Cancel", "إلغاء"),
    "close": ("Close", "إغلاق"),
    "enable": ("Enable", "تمكين"),
    "disable": ("Disable", "تعطيل"),
    "uninstall": ("Uninstall", "إزالة"),
    "test": ("Test", "اختبار"),
    "details": ("Details", "التفاصيل"),
    "refresh": ("Refresh", "تحديث"),
    "clear": ("Clear", "مسح"),
    "save": ("Save", "حفظ"),
    "new": ("New", "جديد"),
    "permission_denied": ("Permission denied", "تم رفض الإذن"),
    "no_data": ("Nothing here yet", "لا يوجد شيء بعد"),
    "sources": ("Sources", "المصادر"),
    "disclaimer": ("Informational only - verify with local authorities.", "للمعلومات فقط - تحقق مع السلطات المحلية."),
    "shadow_preview": ("Shadow preview (no changes made)", "معاينة ظلية (دون تنفيذ تغييرات)"),
    "profile": ("Hardware profile", "ملف الموارد"),
    "theme": ("Theme", "المظهر"),
    "ui_language": ("Interface language", "لغة الواجهة"),
    "tts": ("Speech output", "الإخراج الصوتي"),
    "voice": ("Voice input", "الإدخال الصوتي"),
    "model": ("Model", "النموذج"),
    "no_model": ("No local model loaded", "لا يوجد نموذج محلي محمّل"),
    "fallback": ("Heuristic fallback active", "المحرك الاحتياطي نشط"),
    "skills_studio": ("Skill Studio", "استوديو المهارات"),
    "skill_builder": ("Skill Builder", "باني المهارات"),
    "audit": ("Audit log", "سجل التدقيق"),
    "permissions": ("Permissions", "الأذونات"),
    "network": ("Network", "الشبكة"),
    "storage": ("Storage", "التخزين"),
    "battery": ("Battery", "البطارية"),
    "temperature": ("Temperature", "الحرارة"),
    "ram": ("RAM", "الذاكرة"),
    "cores": ("Cores", "الأنوية"),
    "uptime": ("Uptime", "التشغيل"),
    "emergency": ("Emergency", "الطوارئ"),
    "first_aid": ("First aid", "الإسعافات الأولية"),
    "fire": ("Fire", "حريق"),
    "earthquake": ("Earthquake", "زلزال"),
    "flood": ("Flood", "فيضان"),
    "survival": ("Survival", "البقاء"),
    "numbers": ("Emergency numbers", "أرقام الطوارئ"),
    "verify": ("Verify numbers locally; they can change.", "تحقق محليًا من الأرقام؛ قد تتغير."),
    "retention_days": ("Memory retention (days, 0 = forever)", "الاحتفاظ بالذاكرة (أيام، 0 = دائم)"),
    "wake_word": ("Wake word", "كلمة التنبيه"),
    "continuous": ("Continuous conversation", "محادثة مستمرة"),
    "encryption": ("Encryption at rest", "التشفير عند التخزين"),
    "encryption_unavailable": ("Encryption unavailable on this platform", "التشفير غير متاح على هذه المنصة"),
    "model_import": ("Import GGUF model", "استيراد نموذج GGUF"),
    "delete_all": ("Reset application data", "إعادة ضبط بيانات التطبيق"),
    "run": ("Run", "تشغيل"),
    "pause": ("Pause", "إيقاف مؤقت"),
    "resume": ("Resume", "استئناف"),
    "cancel_run": ("Cancel run", "إلغاء التنفيذ"),
    "report": ("Report", "تقرير"),
    "health": ("Health", "الحالة"),
    "version": ("Version", "الإصدار"),
    "category": ("Category", "الفئة"),
    "risk": ("Risk", "الخطورة"),
    "offline_ok": ("Works offline", "يعمل دون اتصال"),
    "requires_net": ("Requires internet", "يتطلب الإنترنت"),
}

_UI_LANG = "bilingual"


def set_ui_lang(lang: str) -> None:
    global _UI_LANG
    _UI_LANG = lang if lang in ("bilingual", "en", "ar") else "bilingual"


def L(key: str) -> str:
    """Returns logical (unshaped) UI text; widgets reshape exactly once at render time."""
    en, ar = I18N.get(key, (key, key))
    if _UI_LANG == "en":
        return en
    if _UI_LANG == "ar":
        return ar
    return f"{en} · {ar}"


def bilingual(en: str, ar: str) -> str:
    if _UI_LANG == "en":
        return en
    if _UI_LANG == "ar":
        return ar
    return f"{en} · {ar}"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
VALID_THEMES = ("aurora", "matrix", "neon", "lava", "darkblue", "minimal", "glass")
VALID_PROFILES = ("LITE", "BALANCED", "PERFORMANCE")
VALID_MODES = ("local", "online", "auto")
VALID_BUTTON_STYLES = ("glass", "rounded", "pill")
VALID_SPHERE_STYLES = ("liquid", "particles", "plasma")
VALID_SPHERE_COLORS = ("cyan", "magenta", "gold", "green", "violet")
# The living sphere is an independent visual layer: its colours never follow
# the interface theme, so a theme switch cannot repaint the ball.
SPHERE_PAIRS: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "cyan": ((0.15, 0.92, 1.0), (0.24, 0.42, 1.0)),
    "magenta": ((1.0, 0.20, 0.78), (1.0, 0.55, 0.22)),
    "gold": ((1.0, 0.72, 0.20), (1.0, 0.34, 0.10)),
    "green": ((0.25, 1.0, 0.58), (0.05, 0.55, 0.95)),
    "violet": ((0.62, 0.40, 1.0), (0.98, 0.32, 0.72)),
}


def sphere_colors(value: str) -> Tuple[List[float], List[float]]:
    primary, glow = SPHERE_PAIRS.get(value, SPHERE_PAIRS["cyan"])
    return list(primary), list(glow)


def sphere_rgb_int(rgb: Sequence[float]) -> int:
    r, g, b = (int(round(max(0.0, min(1.0, float(value))) * 255)) for value in list(rgb)[:3])
    return (r << 16) | (g << 8) | b


def sphere_style_index(value: str) -> int:
    try:
        return VALID_SPHERE_STYLES.index(value)
    except ValueError:
        return 0


@dataclass
class Config:
    schema: int = 3
    theme: str = "aurora"
    profile: str = "BALANCED"
    ui_lang: str = "bilingual"
    language: str = "auto"           # conversation language: auto|en|ar
    online_mode: str = "auto"        # local|online|auto
    wake_word: str = "hello boshra"
    continuous_conversation: bool = False
    tts_enabled: bool = True
    tts_rate: float = 1.0
    tts_voice: str = "auto"
    piper_voice: str = ""
    piper_voice_lang: str = ""
    ui_sound: bool = True
    ui_sound_volume: float = 0.35
    shader_intensity: float = 1.0
    animations: bool = True
    retention_days: int = 0
    encrypt_storage: bool = True
    allow_python_skills: bool = False
    confirm_high_risk: bool = True
    shadow_by_default: bool = True
    auto_unload_model: bool = True
    selected_model: str = ""
    recording: bool = False
    vault_folders: List[str] = field(default_factory=list)
    onboarding_done: bool = False
    sphere_color: str = "cyan"
    accent_color: str = ""
    button_style: str = "glass"
    sphere_style: str = "liquid"
    assets_path: str = ""
    last_conversation_id: int = 0
    proactive_enabled: bool = True
    morning_hour: int = 8
    evening_hour: int = 20
    quiet_start: str = "22:00"
    quiet_end: str = "07:00"
    commitment_detection: bool = True
    compaction_enabled: bool = True
    auto_backup_days: int = 0
    float_enabled: bool = False
    float_start_on_launch: bool = False
    hijri_dates: bool = True
    arabic_numerals: bool = False
    rtl_layout: bool = True

    def sanitize(self) -> "Config":
        defaults = Config()
        # Config files can be edited or restored from older builds. Validate
        # types before calling string methods or converting numeric values.
        for name in self.__dataclass_fields__:
            default = getattr(defaults, name)
            value = getattr(self, name)
            if isinstance(default, bool):
                if isinstance(value, str):
                    value = value.strip().lower()
                    value = {"true": True, "false": False, "1": True, "0": False}.get(value, default)
                elif isinstance(value, (bool, int)):
                    value = bool(value)
                else:
                    value = default
            elif isinstance(default, (int, float)):
                try:
                    value = type(default)(value)
                    if not math.isfinite(value):
                        value = default
                except (ValueError, TypeError, OverflowError):
                    value = default
            elif isinstance(default, str) and not isinstance(value, str):
                value = default
            elif isinstance(default, list):
                value = value if isinstance(value, list) else []
            setattr(self, name, value)
        choices = {
            "theme": VALID_THEMES, "profile": VALID_PROFILES,
            "online_mode": VALID_MODES, "ui_lang": ("bilingual", "en", "ar"),
            "language": ("auto", "en", "ar"), "button_style": VALID_BUTTON_STYLES,
            "sphere_style": VALID_SPHERE_STYLES, "sphere_color": VALID_SPHERE_COLORS,
            "tts_voice": ("auto", "piper", "android", "desktop"),
            "piper_voice_lang": ("", "ar", "en"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                setattr(self, name, getattr(defaults, name))
        self.tts_rate = clamp(self.tts_rate, 0.5, 2.0)
        self.ui_sound_volume = clamp(self.ui_sound_volume, 0.0, 1.0)
        self.shader_intensity = clamp(self.shader_intensity, 0.2, 1.6)
        self.retention_days = max(0, min(self.retention_days, 3650))
        self.last_conversation_id = max(0, self.last_conversation_id)
        self.morning_hour = max(0, min(int(self.morning_hour), 23))
        self.evening_hour = max(0, min(int(self.evening_hour), 23))
        self.auto_backup_days = max(0, min(int(self.auto_backup_days), 30))
        for name in ("quiet_start", "quiet_end"):
            value = str(getattr(self, name) or "")
            if not re.fullmatch(r"\d{1,2}:\d{2}", value):
                setattr(self, name, "22:00" if name == "quiet_start" else "07:00")
        self.wake_word = (self.wake_word.strip() or "hello boshra")[:48]
        self.piper_voice = self.piper_voice.strip()
        if not self.piper_voice:
            self.piper_voice_lang = ""
        self.vault_folders = [p for p in self.vault_folders if isinstance(p, str)][:20]
        return self


class ConfigManager:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> Config:
        raw = safe_json_load(self.path, {})
        config = Config()
        if isinstance(raw, dict):
            for field_name in config.__dataclass_fields__:
                if field_name in raw:
                    try:
                        setattr(config, field_name, raw[field_name])
                    except (TypeError, ValueError):
                        pass
        config = config.sanitize()
        set_ui_lang(config.ui_lang)
        return config

    def save(self) -> bool:
        with self._lock:
            self.data = self.data.sanitize()
            return atomic_write_text(
                self.path, json.dumps(asdict(self.data), ensure_ascii=False, indent=2)
            )

    def reload(self) -> None:
        with self._lock:
            self.data = self._load()

    def set(self, key: str, value: Any, save: bool = True) -> None:
        with self._lock:
            if key in self.data.__dataclass_fields__:
                setattr(self.data, key, value)
                self.data = self.data.sanitize()
                if save:
                    self.save()

    def __getattr__(self, item: str) -> Any:
        data = object.__getattribute__(self, "data")
        if item in data.__dataclass_fields__:
            return getattr(data, item)
        raise AttributeError(item)


# --------------------------------------------------------------------------
# Encryption at rest (Android Keystore preferred, Fernet on desktop)
# --------------------------------------------------------------------------
class CryptoBackend:
    name = "none"
    available = False
    reason = "not initialised"

    def encrypt(self, text: str) -> str:
        return text

    def decrypt(self, text: str) -> str:
        return text


class FernetBackend(CryptoBackend):
    name = "fernet-file-key"
    available = True
    reason = "cryptography Fernet with a 0600 local key"

    def __init__(self, key_path: Path) -> None:
        from cryptography.fernet import Fernet

        self.key_path = key_path
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.exists():
            key = key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            atomic_write_text(key_path, key.decode("ascii"))
        self._fernet = Fernet(key)

    def encrypt(self, text: str) -> str:
        return "fr:" + self._fernet.encrypt(text.encode("utf-8")).decode("ascii")

    def decrypt(self, text: str) -> str:
        if text.startswith("fr:"):
            return self._fernet.decrypt(text[3:].encode("ascii")).decode("utf-8")
        return text


class AndroidKeystoreBackend(CryptoBackend):
    """AES-256/GCM with a non-exportable Android Keystore key."""

    name = "android-keystore-aes256gcm"
    available = False
    reason = "keystore initialisation pending"
    ALIAS = "boshra_ai_local_key_v1"

    def __init__(self) -> None:
        try:
            from jnius import autoclass  # type: ignore

            self._autoclass = autoclass
            self._KeyStore = autoclass("java.security.KeyStore")
            self._KeyGenerator = autoclass("javax.crypto.KeyGenerator")
            self._Cipher = autoclass("javax.crypto.Cipher")
            self._KeyProperties = autoclass("android.security.keystore.KeyProperties")
            self._KeyGenParameterSpec = autoclass("android.security.keystore.KeyGenParameterSpec")
            self._GCMParameterSpec = autoclass("javax.crypto.spec.GCMParameterSpec")
            self._Base64 = autoclass("android.util.Base64")
            self._key = self._load_or_create_key()
            self.available = True
            self.reason = "Android Keystore AES-256/GCM, key non-exportable"
        except Exception as exc:  # noqa: BLE001
            self.available = False
            self.reason = f"Android Keystore unavailable: {exc}"

    def _load_or_create_key(self):
        store = self._KeyStore.getInstance("AndroidKeyStore")
        store.load(None)
        if not store.containsAlias(self.ALIAS):
            generator = self._KeyGenerator.getInstance("AES", "AndroidKeyStore")
            builder = self._KeyGenParameterSpec.Builder(
                self.ALIAS,
                self._KeyProperties.PURPOSE_ENCRYPT | self._KeyProperties.PURPOSE_DECRYPT,
            )
            builder.setBlockModes(self._KeyProperties.BLOCK_MODE_GCM)
            builder.setEncryptionPaddings(self._KeyProperties.ENCRYPTION_PADDING_NONE)
            builder.setKeySize(256)
            generator.init(builder.build())
            generator.generateKey()
        return store.getKey(self.ALIAS, None)

    def encrypt(self, text: str) -> str:
        cipher = self._Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(self._Cipher.ENCRYPT_MODE, self._key)
        iv = bytes(cipher.getIV())
        payload = iv + bytes(cipher.doFinal(text.encode("utf-8")))
        encoded = self._Base64.encodeToString(payload, self._Base64.NO_WRAP)
        return "ak:" + str(encoded)

    def decrypt(self, text: str) -> str:
        if not text.startswith("ak:"):
            return text
        payload = bytes(self._Base64.decode(text[3:], self._Base64.NO_WRAP))
        iv, body = payload[:12], payload[12:]
        cipher = self._Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(
            self._Cipher.DECRYPT_MODE,
            self._key,
            self._GCMParameterSpec(128, iv),
        )
        return bytes(cipher.doFinal(body)).decode("utf-8")


class CryptoManager:
    def __init__(self, enabled: bool) -> None:
        if not enabled:
            self.backend = CryptoBackend()
            self.backend.available = False
            self.backend.reason = "disabled in settings"
            return
        if IS_ANDROID:
            self.backend = AndroidKeystoreBackend()
        else:
            try:
                self.backend = FernetBackend(PRIVATE_DIR / "secrets" / "db.key")
            except Exception as exc:  # noqa: BLE001
                self.backend = CryptoBackend()
                self.backend.reason = f"Fernet unavailable: {exc}"

    @property
    def available(self) -> bool:
        return bool(self.backend.available)

    @property
    def status(self) -> str:
        return f"{self.backend.name} ({'active' if self.backend.available else 'inactive'}: {self.backend.reason})"

    def seal(self, text: str) -> Tuple[str, int]:
        if not self.available or not text:
            return text, 0
        try:
            return self.backend.encrypt(text), 1
        except Exception as exc:  # noqa: BLE001
            LOG.error("encryption failed", exc)
            return text, 0

    def open(self, text: str, encrypted: int) -> str:
        if not encrypted or not isinstance(text, str):
            return text
        try:
            return self.backend.decrypt(text)
        except Exception as exc:  # noqa: BLE001
            LOG.error("decryption failed", exc)
            return "[encrypted content unavailable]"


# --------------------------------------------------------------------------
# SQLite store
# --------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    convo_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    enc INTEGER NOT NULL DEFAULT 0,
    lang TEXT NOT NULL DEFAULT 'en',
    ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_convo ON messages(convo_id, id);
CREATE TABLE IF NOT EXISTS memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    enc INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'user',
    tags TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category, updated);
CREATE TABLE IF NOT EXISTS memory_vectors(
    memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    path TEXT NOT NULL DEFAULT '',
    sha TEXT NOT NULL DEFAULT '',
    pages INTEGER NOT NULL DEFAULT 0,
    size INTEGER NOT NULL DEFAULT 0,
    added TEXT NOT NULL,
    mime TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL DEFAULT 0,
    ord INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    enc INTEGER NOT NULL DEFAULT 0,
    vec BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ord);
CREATE TABLE IF NOT EXISTS skills(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    risk TEXT NOT NULL DEFAULT 'low',
    offline INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'builtin',
    path TEXT NOT NULL DEFAULT '',
    manifest TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    health TEXT NOT NULL DEFAULT 'healthy',
    last_run TEXT NOT NULL DEFAULT '',
    errors INTEGER NOT NULL DEFAULT 0,
    installed TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skill_grants(
    skill_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    granted INTEGER NOT NULL DEFAULT 0,
    ts TEXT NOT NULL,
    PRIMARY KEY(skill_id, permission)
);
CREATE TABLE IF NOT EXISTS skill_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id TEXT NOT NULL,
    action TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS perms(
    permission TEXT PRIMARY KEY,
    granted INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    category TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    risk TEXT NOT NULL DEFAULT 'low',
    online INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
CREATE TABLE IF NOT EXISTS tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    due TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    enc INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    start TEXT NOT NULL,
    end TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    at TEXT NOT NULL,
    fired INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routines(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    steps TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_state(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    convo_id INTEGER NOT NULL DEFAULT 0,
    goal TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    steps TEXT NOT NULL DEFAULT '[]',
    result TEXT NOT NULL DEFAULT '',
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plans_created ON plans(created);
CREATE TABLE IF NOT EXISTS recall_items(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    ref TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL,
    enc INTEGER NOT NULL DEFAULT 0,
    vec BLOB,
    updated TEXT NOT NULL,
    UNIQUE(kind, ref)
);
CREATE INDEX IF NOT EXISTS idx_recall_kind ON recall_items(kind, updated);
CREATE TABLE IF NOT EXISTS conversation_summaries(
    convo_id INTEGER PRIMARY KEY,
    summary TEXT NOT NULL,
    enc INTEGER NOT NULL DEFAULT 0,
    through_id INTEGER NOT NULL DEFAULT 0,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS commitments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'chat',
    status TEXT NOT NULL DEFAULT 'open',
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_commitments_status ON commitments(status, created);
CREATE TABLE IF NOT EXISTS digests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    day TEXT NOT NULL,
    body TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_digests_day ON digests(day, kind);
"""


class Store:
    """Thread-safe SQLite access with optional field-level encryption."""

    def __init__(self, path: Path = DB_PATH, encrypt: bool = True) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.crypto = CryptoManager(encrypt)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=15.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock, self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute("PRAGMA user_version=3")

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self.lock:
            try:
                rows = self.conn.execute(sql, params).fetchall()
                return [dict(row) for row in rows]
            except sqlite3.Error as exc:
                LOG.error(f"query failed: {sql[:120]}", exc)
                return []

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.lock:
            try:
                with self.conn:
                    cursor = self.conn.execute(sql, params)
                    return int(cursor.lastrowid or 0)
            except sqlite3.Error as exc:
                LOG.error(f"execute failed: {sql[:120]}", exc)
                return 0

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.lock:
            try:
                with self.conn:
                    self.conn.executemany(sql, rows)
            except sqlite3.Error as exc:
                LOG.error(f"executemany failed: {sql[:120]}", exc)

    def seal(self, text: str) -> Tuple[str, int]:
        return self.crypto.seal(text)

    def unseal(self, text: str, encrypted: int) -> str:
        return self.crypto.open(text, encrypted)

    def audit(self, category: str, action: str, detail: str = "", risk: str = "low", online: bool = False) -> None:
        self.execute(
            "INSERT INTO audit(ts, category, action, detail, risk, online) VALUES(?,?,?,?,?,?)",
            (now_iso(), category, action, detail[:4000], risk, 1 if online else 0),
        )

    def state_get(self, key: str, default: str = "") -> str:
        row = self.query_one("SELECT value FROM app_state WHERE key=?", (key,))
        return str(row["value"]) if row else default

    def state_set(self, key: str, value: str) -> None:
        self.execute(
            "INSERT INTO app_state(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def integrity(self) -> Tuple[bool, str]:
        try:
            with self.lock:
                result = self.conn.execute("PRAGMA integrity_check").fetchone()
            ok = bool(result) and str(result[0]).lower() == "ok"
            return ok, str(result[0]) if result else "no result"
        except sqlite3.Error as exc:
            return False, str(exc)

    def vacuum(self) -> bool:
        try:
            with self.lock:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.conn.execute("VACUUM")
            return True
        except sqlite3.Error as exc:
            LOG.error("vacuum failed", exc)
            return False

    def migrate_encryption(self) -> int:
        """Seal legacy plaintext sensitive columns. Returns rows changed."""
        if not self.crypto.available:
            return 0
        changed = 0
        table_specs = [
            ("messages", "content"),
            ("memories", "content"),
            ("notes", "body"),
            ("chunks", "text"),
        ]
        for table, column in table_specs:
            rows = self.query(
                f"SELECT id, {column} AS body FROM {table} WHERE enc=0"
            )
            for row in rows:
                sealed, flag = self.seal(str(row["body"]))
                if flag:
                    self.execute(
                        f"UPDATE {table} SET {column}=?, enc=1 WHERE id=?",
                        (sealed, row["id"]),
                    )
                    changed += 1
        return changed

    def close(self) -> None:
        with contextlib.suppress(Exception):
            with self.lock:
                self.conn.close()


# --------------------------------------------------------------------------
# Lightweight local embeddings (deterministic hashed n-grams, no dependencies)
# --------------------------------------------------------------------------
VECTOR_DIM = 256
_WORD_RE = re.compile(r"[a-z0-9]+|[\u0600-\u06FF]+", re.IGNORECASE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "with", "this", "that", "it", "as", "at", "by", "from",
    "من", "في", "على", "و", "ال", "إلى", "عن", "هذا", "هذه", "ما", "هل", "كيف",
}


def tokenize(text: str) -> List[str]:
    normalized = normalize_arabic(text or "").lower()
    tokens = [t for t in _WORD_RE.findall(normalized) if len(t) > 1 and t not in _STOPWORDS]
    return tokens[:1200]


def embed_text(text: str) -> List[float]:
    vec = [0.0] * VECTOR_DIM

    def add(token: str, weight: float) -> None:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[index] += sign * weight

    tokens = tokenize(text)
    for i, token in enumerate(tokens):
        add("w:" + token, 1.0)
        if i + 1 < len(tokens):
            add("b:" + token + "_" + tokens[i + 1], 0.6)
        for j in range(0, max(1, len(token) - 2)):
            add("c:" + token[j:j + 3], 0.25)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def vec_to_blob(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def blob_to_vec(blob: Optional[bytes]) -> List[float]:
    if not blob:
        return []
    count = len(blob) // 4
    try:
        return list(struct.unpack(f"<{count}f", blob[: count * 4]))
    except struct.error:
        return []


def cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    size = min(len(vec_a), len(vec_b))
    return sum(vec_a[i] * vec_b[i] for i in range(size))


_SENSITIVE_RE = re.compile(
    r"\b(password|passwd|api[_ -]?key|secret|token|otp|one[- ]time|seed phrase|"
    r"cvv|pin code|كلمة المرور|رمز سري|مفتاح سري)\b",
    re.IGNORECASE,
)


def looks_sensitive(text: str) -> bool:
    return bool(_SENSITIVE_RE.search(text or ""))


MEMORY_CATEGORIES = (
    "session", "conversation", "profile", "episodic",
    "semantic", "procedural", "project",
)


class MemoryManager:
    """Seven-layer local memory with hashed semantic retrieval and provenance."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def add(
        self,
        category: str,
        content: str,
        key: str = "",
        source: str = "user",
        tags: str = "",
        allow_sensitive: bool = False,
    ) -> Dict[str, Any]:
        category = category if category in MEMORY_CATEGORIES else "semantic"
        content = (content or "").strip()
        if not content:
            return {"ok": False, "error": "empty memory"}
        if len(content) > 8000:
            content = content[:8000]
        if looks_sensitive(content) and not allow_sensitive:
            return {
                "ok": False,
                "error": "Refused: looks like a credential or secret. Boshra never stores passwords or keys.",
            }
        now = now_iso()
        existing = None
        if key:
            existing = self.store.query_one(
                "SELECT id FROM memories WHERE category=? AND key=? COLLATE NOCASE",
                (category, key),
            )
        sealed, enc = self.store.seal(content)
        if existing:
            memory_id = int(existing["id"])
            self.store.execute(
                "UPDATE memories SET content=?, enc=?, source=?, tags=?, updated=? WHERE id=?",
                (sealed, enc, source, tags, now, memory_id),
            )
        else:
            memory_id = self.store.execute(
                "INSERT INTO memories(category,key,content,enc,source,tags,created,updated) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (category, key, sealed, enc, source, tags, now, now),
            )
        if memory_id:
            self.store.execute(
                "INSERT INTO memory_vectors(memory_id, vec) VALUES(?,?) "
                "ON CONFLICT(memory_id) DO UPDATE SET vec=excluded.vec",
                (memory_id, vec_to_blob(embed_text(content))),
            )
            self.store.audit("memory", "remember", f"{category}:{key or memory_id}", "low")
        return {"ok": bool(memory_id), "id": memory_id}

    def _row(self, row: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "category": row["category"],
            "key": row.get("key", ""),
            "content": self.store.unseal(row["content"], row["enc"]),
            "source": row.get("source", ""),
            "tags": row.get("tags", ""),
            "created": row.get("created", ""),
            "updated": row.get("updated", ""),
            "score": round(score, 4),
        }

    def search(self, query: str, k: int = 6, category: Optional[str] = None) -> List[Dict[str, Any]]:
        params: List[Any] = []
        sql = "SELECT * FROM memories"
        if category:
            sql += " WHERE category=?"
            params.append(category)
        rows = self.store.query(sql + " ORDER BY updated DESC LIMIT 800", params)
        qvec = embed_text(query)
        qwords = set(tokenize(query))
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in rows:
            content = self.store.unseal(row["content"], row["enc"])
            vec_row = self.store.query_one(
                "SELECT vec FROM memory_vectors WHERE memory_id=?", (row["id"],)
            )
            vec = blob_to_vec(vec_row["vec"]) if vec_row else embed_text(content)
            cosine_score = cosine(qvec, vec)
            words = set(tokenize(content))
            overlap = len(qwords & words) / (len(qwords) or 1)
            haystack = normalize_arabic(content).lower()
            keyword_bonus = 0.35 if query and normalize_arabic(query).lower() in haystack else 0.0
            score = cosine_score * 0.7 + overlap * 0.3 + keyword_bonus
            if score > 0.02:
                scored.append((score, self._row(row, score)))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[: max(1, min(k, 50))]]

    def list(self, category: Optional[str] = None, limit: int = 300) -> List[Dict[str, Any]]:
        if category:
            rows = self.store.query(
                "SELECT * FROM memories WHERE category=? ORDER BY updated DESC LIMIT ?",
                (category, limit),
            )
        else:
            rows = self.store.query(
                "SELECT * FROM memories ORDER BY updated DESC LIMIT ?", (limit,)
            )
        return [self._row(row) for row in rows]

    def update(self, memory_id: int, content: str) -> bool:
        if looks_sensitive(content):
            return False
        sealed, enc = self.store.seal(content[:8000])
        self.store.execute(
            "UPDATE memories SET content=?, enc=?, updated=? WHERE id=?",
            (sealed, enc, now_iso(), memory_id),
        )
        self.store.execute(
            "INSERT INTO memory_vectors(memory_id, vec) VALUES(?,?) "
            "ON CONFLICT(memory_id) DO UPDATE SET vec=excluded.vec",
            (memory_id, vec_to_blob(embed_text(content))),
        )
        return True

    def delete(self, memory_id: int) -> bool:
        self.store.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.store.audit("memory", "delete", str(memory_id), "medium")
        return True

    def clear(self, category: Optional[str] = None) -> int:
        before = self.store.query_one("SELECT COUNT(*) AS c FROM memories")
        if category:
            self.store.execute("DELETE FROM memories WHERE category=?", (category,))
        else:
            self.store.execute("DELETE FROM memories")
        self.store.audit("memory", "clear", category or "all", "high")
        after = self.store.query_one("SELECT COUNT(*) AS c FROM memories")
        return int((before or {"c": 0})["c"]) - int((after or {"c": 0})["c"])

    def profile_get(self, key: str, default: str = "") -> str:
        row = self.store.query_one(
            "SELECT content, enc FROM memories WHERE category='profile' AND key=? COLLATE NOCASE",
            (key,),
        )
        return self.store.unseal(row["content"], row["enc"]) if row else default

    def profile_set(self, key: str, value: str) -> bool:
        return bool(self.add("profile", value, key=key, source="user").get("ok"))

    def purge_expired(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat(timespec="seconds")
        rows = self.store.query("SELECT id FROM memories WHERE updated < ?", (cutoff,))
        for row in rows:
            self.store.execute("DELETE FROM memories WHERE id=?", (row["id"],))
        if rows:
            self.store.audit("memory", "retention-purge", str(len(rows)), "medium")
        return len(rows)

    def export_all(self) -> Dict[str, Any]:
        return {
            "version": APP_VERSION,
            "exported": now_iso(),
            "memories": self.list(limit=5000),
        }

    def import_all(self, payload: Dict[str, Any]) -> int:
        count = 0
        for item in payload.get("memories", []) or []:
            result = self.add(
                str(item.get("category", "semantic")),
                str(item.get("content", "")),
                key=str(item.get("key", "")),
                source=str(item.get("source", "import")),
                tags=str(item.get("tags", "")),
            )
            count += 1 if result.get("ok") else 0
        self.store.audit("memory", "import", str(count), "medium")
        return count


# --------------------------------------------------------------------------
# Knowledge Vault
# --------------------------------------------------------------------------
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VAULT_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf", ".docx"}


def ocr_image_file(path: Path) -> str:
    """Optional OCR. Uses an installed tesseract engine; never fakes output."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        return str(pytesseract.image_to_string(Image.open(path)))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "OCR engine unavailable. Install tesseract + pytesseract in the build, "
            f"or import a text-based document. ({exc})"
        )


def extract_pdf(path: Path) -> Tuple[str, int]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pypdf not installed in this build") from exc
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise RuntimeError("PDF is encrypted; decrypt it before importing")
    pages: List[str] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001
            text = ""
        pages.append(f"\n[[page {index + 1}]]\n{text}")
    return "".join(pages), len(reader.pages)


def extract_docx(path: Path) -> Tuple[str, int]:
    try:
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" not in archive.namelist():
                raise RuntimeError("Not a Word .docx document")
            xml_bytes = archive.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Corrupt DOCX container") from exc
    root = ET.fromstring(xml_bytes)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: List[str] = []
    for paragraph in root.iter(ns + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(ns + "t"))
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs), 0


def extract_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-16", "cp1256", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            if path.suffix.lower() == ".csv":
                reader = csv.reader(io.StringIO(text))
                return "\n".join(" | ".join(row) for row in reader)
            return text
        except (UnicodeDecodeError, OSError):
            continue
    raise RuntimeError("Unable to decode text document")


def split_chunks(text: str, limit: int = 1100, overlap: int = 160) -> List[str]:
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]
    chunks: List[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= limit:
            buffer = f"{buffer}\n\n{paragraph}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        if len(paragraph) <= limit:
            buffer = paragraph
            continue
        sentences = re.split(r"(?<=[.!?؟])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= limit:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                while len(sentence) > limit:
                    chunks.append(sentence[:limit])
                    sentence = sentence[limit - overlap:]
                current = sentence
        buffer = current
    if buffer:
        chunks.append(buffer)
    # add small overlap between consecutive chunks for better retrieval
    overlapped: List[str] = []
    for index, chunk in enumerate(chunks):
        prefix = chunks[index - 1][-overlap:] if index > 0 else ""
        overlapped.append((prefix + " " + chunk).strip())
    return [c for c in overlapped if c]


class KnowledgeVault:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.vault_files = PRIVATE_DIR / "vault"
        self.vault_files.mkdir(parents=True, exist_ok=True)

    def import_file(self, path: str | Path, title: str = "") -> Dict[str, Any]:
        source = Path(path)
        if not source.exists() or not source.is_file():
            return {"ok": False, "error": "File not found"}
        suffix = source.suffix.lower()
        if suffix not in VAULT_EXTENSIONS:
            return {"ok": False, "error": f"Unsupported format: {suffix or 'unknown'}"}
        size = source.stat().st_size
        if size > 120 * 1024 * 1024:
            return {"ok": False, "error": "Document exceeds 120 MB limit"}
        digest = sha256_file(source)
        duplicate = self.store.query_one("SELECT id,title FROM documents WHERE sha=?", (digest,))
        if duplicate:
            return {
                "ok": True,
                "doc_id": duplicate["id"],
                "message": f"Already indexed as '{duplicate['title']}'",
                "duplicate": True,
            }
        try:
            if suffix == ".pdf":
                text, pages = extract_pdf(source)
            elif suffix == ".docx":
                text, pages = extract_docx(source)
            elif suffix in IMAGE_EXTENSIONS:
                text, pages = ocr_image_file(source), 1
            else:
                text, pages = extract_text_file(source), 0
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        if not text.strip():
            return {"ok": False, "error": "No extractable text found (scanned PDFs need OCR)"}
        try:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.name)[:80] or "document"
            copied = self.vault_files / f"{digest[:12]}_{safe_name}"
            if not copied.exists():
                shutil.copy2(source, copied)
        except OSError as exc:
            LOG.warning(f"vault copy failed: {exc}")
            copied = source
        doc_title = (title or source.stem).strip()[:160]
        doc_id = self.store.execute(
            "INSERT INTO documents(title,path,sha,pages,size,added,mime) VALUES(?,?,?,?,?,?,?)",
            (doc_title, str(copied), digest, pages, size, now_iso(), suffix.lstrip(".")),
        )
        if not doc_id:
            return {"ok": False, "error": "Database write failed"}
        count = 0
        for index, chunk in enumerate(split_chunks(text)):
            page = 0
            page_markers = re.findall(r"\[\[page (\d+)\]\]", text)
            if page_markers:
                consumed = text.find(chunk[:60])
                if consumed >= 0:
                    pages_before = re.findall(r"\[\[page (\d+)\]\]", text[:consumed])
                    page = int(pages_before[-1]) if pages_before else 1
            sealed, enc = self.store.seal(chunk)
            self.store.execute(
                "INSERT INTO chunks(doc_id,page,ord,text,enc,vec) VALUES(?,?,?,?,?,?)",
                (doc_id, page, index, sealed, enc, vec_to_blob(embed_text(chunk))),
            )
            count += 1
        self.store.audit("vault", "import", f"{doc_title} ({count} chunks)", "low")
        return {
            "ok": True,
            "doc_id": doc_id,
            "chunks": count,
            "pages": pages,
            "message": f"Indexed {doc_title} · {count} chunks",
        }

    def documents(self) -> List[Dict[str, Any]]:
        return self.store.query("SELECT * FROM documents ORDER BY added DESC")

    def delete_document(self, doc_id: int) -> bool:
        row = self.store.query_one("SELECT path FROM documents WHERE id=?", (doc_id,))
        self.store.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        if row:
            path = Path(row["path"])
            if is_within(path, self.vault_files):
                with contextlib.suppress(OSError):
                    path.unlink()
        self.store.audit("vault", "delete-document", str(doc_id), "high")
        return True

    def _chunk_rows(self, doc_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if doc_id:
            return self.store.query(
                "SELECT c.*, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id "
                "WHERE c.doc_id=? ORDER BY c.ord",
                (doc_id,),
            )
        return self.store.query(
            "SELECT c.*, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "ORDER BY c.doc_id, c.ord LIMIT 5000"
        )

    def search(self, query: str, k: int = 6, doc_id: Optional[int] = None) -> List[Dict[str, Any]]:
        qvec = embed_text(query)
        qwords = set(tokenize(query))
        results: List[Tuple[float, Dict[str, Any]]] = []
        for row in self._chunk_rows(doc_id):
            text = self.store.unseal(row["text"], row["enc"])
            vec = blob_to_vec(row.get("vec")) or embed_text(text)
            semantic = cosine(qvec, vec)
            words = set(tokenize(text))
            overlap = len(qwords & words) / (len(qwords) or 1)
            score = semantic * 0.65 + overlap * 0.35
            if score > 0.05:
                results.append(
                    (
                        score,
                        {
                            "chunk_id": row["id"],
                            "doc_id": row["doc_id"],
                            "title": row["title"],
                            "page": row["page"],
                            "text": text,
                            "score": round(score, 4),
                        },
                    )
                )
        results.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in results[: max(1, min(k, 30))]]

    def document_text(self, doc_id: int) -> str:
        return "\n\n".join(
            self.store.unseal(row["text"], row["enc"]) for row in self._chunk_rows(doc_id)
        )

    def answer(self, question: str, k: int = 4) -> Dict[str, Any]:
        passages = self.search(question, k=k)
        if not passages:
            return {
                "ok": False,
                "error": "No matching passages in the Knowledge Vault",
                "sources": [],
            }
        terms = set(tokenize(question))
        sentences: List[Tuple[float, str, Dict[str, Any]]] = []
        for passage in passages:
            for sentence in re.split(r"(?<=[.!?؟])\s+", passage["text"]):
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue
                overlap = len(terms & set(tokenize(sentence))) / (len(terms) or 1)
                sentences.append((overlap + passage["score"], sentence, passage))
        sentences.sort(key=lambda item: item[0], reverse=True)
        chosen: List[str] = []
        used_chunks: Dict[int, Dict[str, Any]] = {}
        for _, sentence, passage in sentences[:8]:
            if sentence not in chosen:
                chosen.append(sentence)
            used_chunks[passage["chunk_id"]] = passage
            if len(chosen) >= 4:
                break
        sources = [
            {"title": p["title"], "page": p["page"], "doc_id": p["doc_id"]}
            for p in list(used_chunks.values())[:4]
        ]
        body = " ".join(chosen)
        return {"ok": True, "answer": body, "sources": sources, "passages": passages}

    def summarize(self, doc_id: int, max_sentences: int = 6) -> Dict[str, Any]:
        rows = self._chunk_rows(doc_id)
        if not rows:
            return {"ok": False, "error": "Document not indexed"}
        full = "\n".join(self.store.unseal(row["text"], row["enc"]) for row in rows)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", full) if len(s.strip()) > 40]
        if not sentences:
            return {"ok": False, "error": "Not enough prose to summarize"}
        frequency: Dict[str, int] = {}
        for word in tokenize(full):
            frequency[word] = frequency.get(word, 0) + 1
        ranked = sorted(
            sentences,
            key=lambda s: sum(frequency.get(w, 0) for w in set(tokenize(s))) / (len(tokenize(s)) ** 0.5 or 1),
            reverse=True,
        )
        chosen = sorted(set(ranked[:max_sentences]), key=sentences.index)
        return {
            "ok": True,
            "summary": "\n".join(f"• {s}" for s in chosen),
            "title": rows[0]["title"],
            "sentences": len(sentences),
        }

    def export_summary(self, doc_id: int, max_sentences: int = 8) -> Dict[str, Any]:
        result = self.summarize(doc_id, max_sentences)
        if not result.get("ok"):
            return result
        title = str(result.get("title", "document"))
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:60] or "document"
        out = EXPORT_DIR / f"{safe}_summary.md"
        body = (
            f"# {title}\n\n_Generated locally by {APP_NAME} {APP_VERSION} on {now_iso()}_\n\n"
            f"{result['summary']}\n\n---\nSources: {title}\n"
        )
        if not atomic_write_text(out, body, 0o644):
            return {"ok": False, "error": "Could not write export file"}
        self.store.audit("vault", "export-summary", str(out), "low")
        return {"ok": True, "path": str(out), "summary": result["summary"]}


# --------------------------------------------------------------------------
# Android bridge (Pyjnius) with honest desktop fallbacks
# --------------------------------------------------------------------------
class AndroidBridge:
    def __init__(self) -> None:
        self.available = False
        self.reason = "desktop mode"
        self.activity = None
        self.context = None
        if not IS_ANDROID:
            return
        try:
            from jnius import autoclass  # type: ignore

            self.autoclass = autoclass
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            self.activity = PythonActivity.mActivity
            self.context = self.activity.getApplicationContext()
            self.available = bool(self.context)
            self.reason = "Pyjnius bridge active" if self.available else "no activity context"
        except Exception as exc:  # noqa: BLE001
            self.reason = f"Pyjnius bridge unavailable: {exc}"
            LOG.error("android bridge init failed", exc)

    # -- sensors / device ------------------------------------------------
    def play_click(self, volume: float = 0.4) -> bool:
        """System key-click used only when Kivy audio cannot play our tone."""
        if not self.available:
            return False
        try:
            Context = self.autoclass("android.content.Context")
            AudioManager = self.autoclass("android.media.AudioManager")
            manager = self.context.getSystemService(Context.AUDIO_SERVICE)
            if not manager:
                return False
            manager.playSoundEffect(
                AudioManager.FX_KEY_CLICK, float(clamp(volume, 0.0, 1.0))
            )
            return True
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"system click failed: {exc}")
            return False

    def vibrate(self, milliseconds: int = 40) -> bool:
        if not self.available:
            return False
        try:
            Context = self.autoclass("android.content.Context")

            vibrator = self.context.getSystemService(Context.VIBRATOR_SERVICE)
            if not vibrator or not vibrator.hasVibrator():
                return False
            try:
                VibrationEffect = self.autoclass("android.os.VibrationEffect")
                effect = VibrationEffect.createOneShot(int(milliseconds), VibrationEffect.DEFAULT_AMPLITUDE)
                vibrator.vibrate(effect)
            except Exception:
                vibrator.vibrate(int(milliseconds))
            return True
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"vibrate failed: {exc}")
            return False

    def device_info(self) -> Dict[str, Any]:
        if not self.available:
            import platform as _platform

            return {
                "platform": "desktop",
                "model": _platform.machine(),
                "manufacturer": _platform.system(),
                "android": "",
                "sdk": "",
                "cpu": _platform.processor() or _platform.machine(),
                "cores": os.cpu_count() or 0,
            }
        try:
            Build = self.autoclass("android.os.Build")
            BuildVersion = self.autoclass("android.os.Build$VERSION")
            return {
                "platform": "android",
                "model": str(Build.MODEL),
                "manufacturer": str(Build.MANUFACTURER),
                "device": str(Build.DEVICE),
                "android": str(BuildVersion.RELEASE),
                "sdk": int(BuildVersion.SDK_INT),
                "cpu": str(Build.SUPPORTED_ABIS[0]) if Build.SUPPORTED_ABIS else "",
                "cores": os.cpu_count() or 0,
            }
        except Exception as exc:  # noqa: BLE001
            return {"platform": "android", "error": str(exc), "cores": os.cpu_count() or 0}

    def battery_info(self) -> Dict[str, Any]:
        if not self.available:
            try:
                import psutil  # type: ignore

                battery = psutil.sensors_battery()
                if battery:
                    return {
                        "percent": round(float(battery.percent), 1),
                        "charging": bool(battery.power_plugged),
                        "temperature_c": None,
                    }
            except Exception:
                pass
            return {"percent": None, "charging": None, "temperature_c": None}
        try:
            Intent = self.autoclass("android.content.Intent")
            IntentFilter = self.autoclass("android.content.IntentFilter")
            BatteryManager = self.autoclass("android.os.BatteryManager")
            intent = self.context.registerReceiver(
                None, IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            )
            if intent is None:
                return {"percent": None, "charging": None, "temperature_c": None}
            level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, 100) or 100
            status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            temp = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1)
            return {
                "percent": round(level * 100.0 / scale, 1) if level >= 0 else None,
                "charging": status in (2, 5),  # CHARGING / FULL
                "temperature_c": round(temp / 10.0, 1) if temp > 0 else None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"percent": None, "charging": None, "error": str(exc)}

    def memory_info(self) -> Dict[str, Any]:
        if not self.available:
            try:
                import psutil  # type: ignore

                mem = psutil.virtual_memory()
                return {"total": int(mem.total), "available": int(mem.available)}
            except Exception:
                return {"total": 0, "available": 0}
        try:
            Context = self.autoclass("android.content.Context")
            MemoryInfo = self.autoclass("android.app.ActivityManager$MemoryInfo")
            manager = self.context.getSystemService(Context.ACTIVITY_SERVICE)
            info = MemoryInfo()
            manager.getMemoryInfo(info)
            return {"total": int(info.totalMem), "available": int(info.availMem)}
        except Exception as exc:  # noqa: BLE001
            return {"total": 0, "available": 0, "error": str(exc)}

    def storage_info(self) -> Dict[str, Any]:
        candidate = "/data" if self.available else str(Path.home())
        try:
            usage = shutil.disk_usage(candidate)
            return {"total": usage.total, "free": usage.free}
        except OSError:
            return {"total": 0, "free": 0}

    def thermal_status(self) -> str:
        if not self.available:
            return "desktop"
        try:
            Context = self.autoclass("android.content.Context")
            manager = self.context.getSystemService(Context.POWER_SERVICE)
            status = int(manager.getCurrentThermalStatus())
            return {
                0: "none", 1: "light", 2: "moderate", 3: "severe",
                4: "critical", 5: "emergency", 6: "shutdown",
            }.get(status, str(status))
        except Exception:
            return "unavailable"

    def network_info(self) -> Dict[str, Any]:
        if not self.available:
            info = {"online": False, "type": "desktop", "metered": None}
            try:
                import socket

                socket.create_connection(("1.1.1.1", 53), timeout=1.2).close()
                info["online"] = True
            except OSError:
                pass
            return info
        try:
            Context = self.autoclass("android.content.Context")
            NetworkCapabilities = self.autoclass("android.net.NetworkCapabilities")
            manager = self.context.getSystemService(Context.CONNECTIVITY_SERVICE)
            network = manager.getActiveNetwork()
            if network is None:
                return {"online": False, "type": "none", "metered": None}
            caps = manager.getNetworkCapabilities(network)
            online = bool(caps and caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET))
            metered = bool(caps and not caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED))
            transport = "unknown"
            if caps:
                if caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI):
                    transport = "wifi"
                elif caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR):
                    transport = "cellular"
                elif caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET):
                    transport = "ethernet"
            return {"online": online, "type": transport, "metered": metered}
        except Exception as exc:  # noqa: BLE001
            return {"online": False, "type": "unknown", "error": str(exc)}

    # -- intents ---------------------------------------------------------
    KNOWN_APPS = {
        "whatsapp": "com.whatsapp",
        "واتساب": "com.whatsapp",
        "youtube": "com.google.android.youtube",
        "يوتيوب": "com.google.android.youtube",
        "telegram": "org.telegram.messenger",
        "تلغرام": "org.telegram.messenger",
        "chrome": "com.android.chrome",
        "كروم": "com.android.chrome",
        "gmail": "com.google.android.gm",
        "maps": "com.google.android.apps.maps",
        "خرائط": "com.google.android.apps.maps",
        "camera": "com.android.camera2",
        "كاميرا": "com.android.camera2",
        "phone": "com.android.dialer",
        "هاتف": "com.android.dialer",
        "settings": "com.android.settings",
        "اعدادات": "com.android.settings",
        "files": "com.android.documentsui",
        "ملفات": "com.android.documentsui",
        "calculator": "com.android.calculator2",
        "آلة حاسبة": "com.android.calculator2",
        "facebook": "com.facebook.katana",
        "instagram": "com.instagram.android",
        "spotify": "com.spotify.music",
        "vlc": "org.videolan.vlc",
        "notes": "com.google.android.keep",
        "keep": "com.google.android.keep",
    }

    def open_app(self, query: str) -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "App launching requires Android"}
        query = (query or "").strip().lower()
        try:
            Intent = self.autoclass("android.content.Intent")
            package_manager = self.activity.getPackageManager()
            candidates: List[str] = []
            for alias, package in self.KNOWN_APPS.items():
                if alias in query or query in alias:
                    candidates.append(package)
            if re.fullmatch(r"[A-Za-z0-9_.]+", query) and "." in query:
                candidates.insert(0, query)
            candidates.extend(["com.android.camera2", "com.android.camera"])
            for package in dict.fromkeys(candidates):
                intent = package_manager.getLaunchIntentForPackage(package)
                if intent is not None:
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    self.activity.startActivity(intent)
                    return {"ok": True, "app": package, "message": f"Launched {package}"}
            # fall back to installed launcher labels
            intent = Intent(Intent.ACTION_MAIN)
            intent.addCategory(Intent.CATEGORY_LAUNCHER)
            infos = package_manager.queryIntentActivities(intent, 0)
            for index in range(infos.size()):
                info = infos.get(index)
                label = str(info.loadLabel(package_manager)).lower()
                if query and query in label:
                    target = info.activityInfo.packageName
                    launch = package_manager.getLaunchIntentForPackage(str(target))
                    if launch is not None:
                        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        self.activity.startActivity(launch)
                        return {"ok": True, "app": str(target), "message": f"Launched {label}"}
            return {"ok": False, "error": f"No installed app matched '{query}'"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Intent failed: {exc}"}

    def notify(self, title: str, message: str) -> bool:
        if not self.available:
            return False
        try:
            Context = self.autoclass("android.content.Context")
            NotificationManager = self.autoclass("android.app.NotificationManager")
            NotificationChannel = self.autoclass("android.app.NotificationChannel")
            NotificationBuilder = self.autoclass("android.app.Notification$Builder")
            manager = self.context.getSystemService(Context.NOTIFICATION_SERVICE)
            channel_id = "boshra"
            channel = NotificationChannel(
                channel_id, "Boshra", NotificationManager.IMPORTANCE_DEFAULT
            )
            manager.createNotificationChannel(channel)
            builder = NotificationBuilder(self.context, channel_id)
            builder.setContentTitle(str(title))
            builder.setContentText(str(message))
            builder.setSmallIcon(self.context.getApplicationInfo().icon)
            manager.notify(int(time.time()) % 100000, builder.build())
            return True
        except Exception as exc:  # noqa: BLE001
            LOG.error("notification failed", exc)
            return False

    def clipboard_get(self) -> str:
        if not self.available:
            if IS_DESKTOP:
                try:
                    from kivy.core.clipboard import Clipboard
                    return str(Clipboard.paste() or "")
                except Exception as exc:
                    LOG.debug(f"desktop clipboard read failed: {exc}")
            return ""
        try:
            Context = self.autoclass("android.content.Context")
            manager = self.context.getSystemService(Context.CLIPBOARD_SERVICE)
            clip = manager.getPrimaryClip()
            if clip is None or clip.getItemCount() == 0:
                return ""
            return str(clip.getItemAt(0).getText() or "")
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"clipboard read failed: {exc}")
            return ""

    def clipboard_set(self, text: str) -> bool:
        if not self.available:
            if IS_DESKTOP:
                try:
                    from kivy.core.clipboard import Clipboard
                    Clipboard.copy(str(text))
                    return True
                except Exception as exc:
                    LOG.debug(f"desktop clipboard write failed: {exc}")
            return False
        try:
            Context = self.autoclass("android.content.Context")
            ClipData = self.autoclass("android.content.ClipData")
            manager = self.context.getSystemService(Context.CLIPBOARD_SERVICE)
            manager.setPrimaryClip(ClipData.newPlainText("Boshra", str(text)))
            return True
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"clipboard write failed: {exc}")
            return False

    def ensure_permission(self, permission: str, timeout: float = 45.0) -> bool:
        if not self.available:
            return True
        try:
            from android.permissions import (  # type: ignore
                Permission,
                check_permission,
                request_permissions,
            )

            name = getattr(Permission, permission, permission)
            if check_permission(name):
                return True
            finished = threading.Event()
            result_holder = {"granted": False}

            def callback(requested, grants):
                result_holder["granted"] = bool(grants and grants[0])
                finished.set()


            # request_permissions blocks the UI thread on some builds; run callback-based.
            request_permissions([name], callback)
            finished.wait(timeout)
            return bool(result_holder["granted"])
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"permission request failed: {exc}")
            return False

    def pick_file(self) -> Optional[str]:
        try:
            from plyer import filechooser  # type: ignore

            selection = filechooser.open_file(
                filters=[["Documents", "*.pdf", "*.docx", "*.txt", "*.csv", "*.md", "*.zip"],
                         ["Models", "*.gguf"]]
            )
            if selection:
                return str(selection[0])
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"file chooser failed: {exc}")
        return None

    # -- window insets / responsive shell -------------------------------
    def window_insets(self) -> Dict[str, int]:
        """Return safe-area and system bar sizes in pixels (0 on desktop)."""
        if not self.available:
            return {"left": 0, "top": 0, "right": 0, "bottom": 0,
                    "status_bar": 0, "nav_bar": 0}
        try:
            BuildVersion = self.autoclass("android.os.Build$VERSION")
            resources = self.context.getResources()
            identifier = resources.getIdentifier("status_bar_height", "dimen", "android")
            status_bar = int(resources.getDimensionPixelSize(identifier)) if identifier else 0
            nav_id = resources.getIdentifier("navigation_bar_height", "dimen", "android")
            nav_bar = int(resources.getDimensionPixelSize(nav_id)) if nav_id else 0
            insets = {"left": 0, "top": status_bar, "right": 0, "bottom": nav_bar,
                      "status_bar": status_bar, "nav_bar": nav_bar}
            if int(BuildVersion.SDK_INT) >= 30:
                with contextlib.suppress(Exception):
                    view = self.activity.getWindow().getDecorView()
                    root = view.getRootWindowInsets()
                    if root is not None:
                        insets["left"] = int(root.getSystemWindowInsetLeft())
                        insets["top"] = int(root.getSystemWindowInsetTop())
                        insets["right"] = int(root.getSystemWindowInsetRight())
                        insets["bottom"] = int(root.getSystemWindowInsetBottom())
            return insets
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"window insets unavailable: {exc}")
            return {"left": 0, "top": 0, "right": 0, "bottom": 0,
                    "status_bar": 0, "nav_bar": 0}

    # -- floating mini assistant (foreground overlay service) -----------
    def overlay_granted(self) -> bool:
        if not self.available:
            return False
        try:
            Settings = self.autoclass("android.provider.Settings")
            return bool(Settings.canDrawOverlays(self.context))
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"overlay permission check failed: {exc}")
            return False

    def request_overlay_permission(self) -> bool:
        if not self.available:
            return False
        try:
            Intent = self.autoclass("android.content.Intent")
            Uri = self.autoclass("android.net.Uri")
            intent = Intent(
                "android.settings.action.MANAGE_OVERLAY_PERMISSION",
                Uri.parse(f"package:{self.context.getPackageName()}"),
            )
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            self.context.startActivity(intent)
            return True
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"overlay permission intent failed: {exc}")
            return False

    def float_status(self) -> Dict[str, Any]:
        if not self.available:
            return {"supported": False, "granted": False, "running": False,
                    "suppressed": False, "reason": "desktop mode"}
        try:
            service = self.autoclass("org.bochra.bochra.BoshraFloatService")
        except Exception:
            return {"supported": False, "granted": self.overlay_granted(),
                    "running": False, "suppressed": False,
                    "reason": "float service not packaged"}
        running = False
        suppressed = False
        with contextlib.suppress(Exception):
            running = bool(service.isRunning())
        with contextlib.suppress(Exception):
            suppressed = bool(service.isSuppressed())
        return {"supported": True, "granted": self.overlay_granted(),
                "running": running, "suppressed": suppressed, "reason": ""}

    def float_clear_suppressed(self) -> None:
        if not self.available:
            return
        with contextlib.suppress(Exception):
            service = self.autoclass("org.bochra.bochra.BoshraFloatService")
            service.clearSuppressed()

    def float_start(self, size_dp: int = 58, color: str = "cyan",
                    style: str = "liquid") -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "The floating orb requires Android"}
        if not self.overlay_granted():
            return {"ok": False, "error": "overlay permission required",
                    "needs_permission": True}
        self.float_clear_suppressed()
        try:
            Intent = self.autoclass("android.content.Intent")
            service_class = self.autoclass("org.bochra.bochra.BoshraFloatService")
            intent = Intent(self.context, service_class)
            intent.setAction("org.bochra.bochra.SHOW")
            intent.putExtra("bubble_size", int(size_dp))
            primary, glow = sphere_colors(color)
            intent.putExtra("sphere_primary", int(sphere_rgb_int(primary)))
            intent.putExtra("sphere_glow", int(sphere_rgb_int(glow)))
            intent.putExtra("sphere_style", int(sphere_style_index(style)))
            BuildVersion = self.autoclass("android.os.Build$VERSION")
            if int(BuildVersion.SDK_INT) >= 26:
                self.context.startForegroundService(intent)
            else:
                self.context.startService(intent)
            return {"ok": True, "message": "Floating assistant started"}
        except Exception as exc:  # noqa: BLE001
            LOG.error("float start failed", exc)
            return {"ok": False, "error": f"Could not start the float service: {exc}"}

    def float_refresh(self, color: str = "cyan", style: str = "liquid",
                      size_dp: int = 58) -> None:
        """Re-send bubble styling to a running overlay without restarting it."""
        if not self.available:
            return
        try:
            service_class = self.autoclass("org.bochra.bochra.BoshraFloatService")
            if not bool(service_class.isRunning()):
                return
            Intent = self.autoclass("android.content.Intent")
            intent = Intent(self.context, service_class)
            intent.setAction("org.bochra.bochra.SHOW")
            intent.putExtra("bubble_size", int(size_dp))
            primary, glow = sphere_colors(color)
            intent.putExtra("sphere_primary", int(sphere_rgb_int(primary)))
            intent.putExtra("sphere_glow", int(sphere_rgb_int(glow)))
            intent.putExtra("sphere_style", int(sphere_style_index(style)))
            BuildVersion = self.autoclass("android.os.Build$VERSION")
            if int(BuildVersion.SDK_INT) >= 26:
                self.context.startForegroundService(intent)
            else:
                self.context.startService(intent)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"float refresh failed: {exc}")

    def float_stop(self) -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "The floating orb requires Android"}
        try:
            service = self.autoclass("org.bochra.bochra.BoshraFloatService")
            if not bool(service.isRunning()):
                return {"ok": True, "message": "Floating assistant already hidden"}
            Intent = self.autoclass("android.content.Intent")
            intent = Intent(self.context, service)
            intent.setAction("org.bochra.bochra.HIDE")
            self.context.startService(intent)
            return {"ok": True, "message": "Floating assistant hidden"}
        except Exception as exc:  # noqa: BLE001
            LOG.error("float stop failed", exc)
            return {"ok": False, "error": f"Could not stop the float service: {exc}"}

    def launch_action(self) -> str:
        """Action carried by the intent that launched the running activity."""
        if not self.available:
            return ""
        try:
            intent = self.activity.getIntent()
            if intent is None:
                return ""
            action = str(intent.getStringExtra("boshra_action") or "")
            if action:
                with contextlib.suppress(Exception):
                    intent.removeExtra("boshra_action")
            return action
        except Exception:
            return ""


# --------------------------------------------------------------------------
# Resource governor
# --------------------------------------------------------------------------
PROFILE_SETTINGS: Dict[str, Dict[str, Any]] = {
    "LITE": {
        "n_ctx": 1024, "n_threads": 2, "max_tokens": 220,
        "shader_fps": 24, "retrieval_k": 3, "max_vault_chunks": 250,
    },
    "BALANCED": {
        "n_ctx": 2048, "n_threads": 4, "max_tokens": 380,
        "shader_fps": 40, "retrieval_k": 4, "max_vault_chunks": 800,
    },
    "PERFORMANCE": {
        "n_ctx": 4096, "n_threads": max(4, (os.cpu_count() or 4) - 1),
        "max_tokens": 640, "shader_fps": 60, "retrieval_k": 6,
        "max_vault_chunks": 2000,
    },
}


class ResourceManager:
    def __init__(self, bridge: AndroidBridge, config: ConfigManager) -> None:
        self.bridge = bridge
        self.config = config
        self._started = time.time()
        self._last_measure: Dict[str, Any] = {}
        self._last_measure_at = 0.0

    @property
    def profile(self) -> Dict[str, Any]:
        return PROFILE_SETTINGS.get(self.config.profile, PROFILE_SETTINGS["BALANCED"])

    def measure(self, force: bool = False) -> Dict[str, Any]:
        if not force and time.time() - self._last_measure_at < 1.5:
            return self._last_measure
        memory = self.bridge.memory_info()
        storage = self.bridge.storage_info()
        battery = self.bridge.battery_info()
        total = int(memory.get("total") or 0)
        available = int(memory.get("available") or 0)
        self._last_measure = {
            "ram_total": total,
            "ram_available": available,
            "ram_percent": round((1.0 - available / total) * 100, 1) if total else 0.0,
            "storage_total": int(storage.get("total") or 0),
            "storage_free": int(storage.get("free") or 0),
            "battery": battery,
            "thermal": self.bridge.thermal_status(),
            "cores": os.cpu_count() or 1,
            "network": self.bridge.network_info(),
            "uptime": int(time.time() - self._started),
            "profile": self.config.profile,
        }
        self._last_measure_at = time.time()
        return self._last_measure

    def should_unload_model(self) -> bool:
        if not self.config.auto_unload_model:
            return False
        measure = self.measure(force=True)
        total = measure.get("ram_total") or 0
        available = measure.get("ram_available") or 0
        if total and available and available / total < 0.12:
            return True
        return measure.get("thermal") in ("severe", "critical", "emergency", "shutdown")


# --------------------------------------------------------------------------
# Local LLM: model discovery and inference
# --------------------------------------------------------------------------
MODEL_SCAN_DIRS = [
    MODELS_DIR,
    PUBLIC_DIR / "models",
    ASSETS_DIR / "phi3",
    ASSETS_PUBLIC_DIR / "phi3",
    Path("/sdcard/phi3"),
    Path("/sdcard/assets/phi3"),
    Path("/sdcard/Download/BoshraModels"),
    Path("/sdcard/BoshraAI/models"),
]
if IS_DESKTOP:
    MODEL_SCAN_DIRS.extend([Path.home() / "models", Path.home() / "Downloads"])

_QUANT_RE = re.compile(r"(Q\d[^.\s_-]*|IQ\d[^.\s_-]*|F16|F32|BF16)", re.IGNORECASE)


@dataclass
class ModelInfo:
    name: str
    path: str
    size: int
    modified: float
    quant: str
    magic_ok: bool
    sha: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelManager:
    def __init__(self, config: ConfigManager, store: Store) -> None:
        self.config = config
        self.store = store
        self._scan_cache: List[ModelInfo] = []
        self._last_scan = 0.0

    def scan(self, force: bool = False) -> List[ModelInfo]:
        if not force and time.time() - self._last_scan < 20 and self._scan_cache:
            return self._scan_cache
        found: Dict[str, ModelInfo] = {}
        scan_dirs = list(MODEL_SCAN_DIRS)
        saved_assets = getattr(self.config, "assets_path", "")
        if saved_assets:
            saved_root = Path(saved_assets)
            scan_dirs = [saved_root / "phi3", saved_root] + scan_dirs
        for directory in scan_dirs:
            try:
                if not directory.exists():
                    continue
                for path in list(directory.glob("*.gguf")) + list(directory.glob("*/*.gguf")):
                    if not path.is_file():
                        continue
                    try:
                        stat = path.stat()
                        with path.open("rb") as fh:
                            magic = fh.read(4)
                        match = _QUANT_RE.search(path.name)
                        key = str(path.resolve())
                        found[key] = ModelInfo(
                            name=path.name,
                            path=key,
                            size=stat.st_size,
                            modified=stat.st_mtime,
                            quant=(match.group(1).upper() if match else "unknown"),
                            magic_ok=magic == b"GGUF",
                        )
                    except OSError:
                        continue
            except OSError:
                continue
        self._scan_cache = sorted(found.values(), key=lambda m: m.size, reverse=True)
        self._last_scan = time.time()
        return self._scan_cache

    def selected(self) -> Optional[ModelInfo]:
        selected = self.config.selected_model
        for model in self.scan():
            if model.path == selected:
                return model
        if self._scan_cache:
            return self._scan_cache[0]
        return None

    def select(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        self.config.set("selected_model", path)
        self.store.audit("model", "select", Path(path).name, "low")
        return True

    def verify(self, path: str) -> Dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return {"ok": False, "error": "Model file missing"}
        size = file_path.stat().st_size
        if size < 20 * 1024 * 1024:
            return {"ok": False, "error": f"File too small for a GGUF model ({human_bytes(size)})"}
        try:
            with file_path.open("rb") as fh:
                magic = fh.read(4)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        if magic != b"GGUF":
            return {"ok": False, "error": "Invalid GGUF magic bytes"}
        digest = sha256_file(file_path)
        return {
            "ok": True,
            "sha256": digest,
            "size": size,
            "message": f"Valid GGUF · {human_bytes(size)}",
        }

    def import_model(self, source_path: str) -> Dict[str, Any]:
        source = Path(source_path)
        if not source.exists() or source.suffix.lower() != ".gguf":
            return {"ok": False, "error": "Select a .gguf model file"}
        verified = self.verify(str(source))
        if not verified.get("ok"):
            return verified
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        destination = MODELS_DIR / source.name
        if source.resolve() != destination.resolve():
            if destination.exists():
                existing = self.verify(str(destination))
                if existing.get("sha256") == verified.get("sha256"):
                    return {"ok": True, "path": str(destination), "message": "Model already imported"}
                stamp = datetime.now().strftime("%Y%m%d%H%M%S")
                destination = MODELS_DIR / f"{source.stem}_{stamp}.gguf"
            try:
                with source.open("rb") as src, destination.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            except OSError as exc:
                return {"ok": False, "error": f"Copy failed: {exc}"}
        self.select(str(destination))
        self.scan(force=True)
        self.store.audit("model", "import", destination.name, "medium")
        return {"ok": True, "path": str(destination), "message": f"Imported {destination.name}"}


class LlamaCppEngine:
    """llama.cpp chat engine. Requires an Android-native llama_cpp build."""
    name = "llama.cpp"

    def __init__(self, model_path: str, profile: Dict[str, Any]) -> None:
        self.model_path = model_path
        self.profile = profile
        self.available = False
        self.status = "not loaded"
        self._model = None
        self._lock = threading.RLock()

    def load(self) -> bool:
        with self._lock:
            if self._model is not None:
                self.available = True
                return True
            try:
                from llama_cpp import Llama  # type: ignore

                started = time.time()
                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=int(self.profile["n_ctx"]),
                    n_threads=int(self.profile["n_threads"]),
                    n_batch=128,
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,
                )
                self.available = True
                self.status = f"loaded in {time.time() - started:.1f}s · ctx {self.profile['n_ctx']}"
                LOG.info(f"LLM loaded: {Path(self.model_path).name} {self.status}")
                return True
            except Exception as exc:  # noqa: BLE001
                self._model = None
                self.available = False
                self.status = f"llama.cpp unavailable: {exc}"
                LOG.error("llama load failed", exc)
                return False

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self.available = False
            self.status = "unloaded"
            gc.collect()

    def generate_stream(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        if not self.load():
            raise RuntimeError(self.status)
        assert self._model is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            stream = self._model.create_chat_completion(
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True,
            )
            for event in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                try:
                    delta = event["choices"][0].get("delta", {})
                    token = delta.get("content") or ""
                except (KeyError, IndexError, TypeError):
                    token = ""
                if token:
                    yield token
        except Exception as exc:  # noqa: BLE001
            LOG.error("generation failed", exc)
            raise RuntimeError(f"Generation failed: {exc}") from exc


class HeuristicEngine:
    """Deterministic local fallback. Not an LLM; never pretends to be one."""

    name = "heuristic-offline"

    def __init__(self) -> None:
        self.available = True
        self.status = "extractive + rule engine"

    def load(self) -> bool:
        return True

    def unload(self) -> None:
        return

    def _math(self, text: str) -> Optional[str]:
        import ast

        match = re.search(
            r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/%^]\s*-?\d+(?:\.\d+)?)+)", text
        )
        if not match:
            return None
        expression = match.group(1).replace("^", "**")
        try:
            node = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        def evaluate(current: ast.AST) -> float:
            if isinstance(current, ast.Expression):
                return evaluate(current.body)
            if isinstance(current, ast.Constant) and isinstance(current.value, (int, float)):
                return float(current.value)
            if isinstance(current, ast.UnaryOp) and isinstance(current.op, ast.USub):
                return -evaluate(current.operand)
            if isinstance(current, ast.BinOp):
                left, right = evaluate(current.left), evaluate(current.right)
                if isinstance(current.op, ast.Add):
                    return left + right
                if isinstance(current.op, ast.Sub):
                    return left - right
                if isinstance(current.op, ast.Mult):
                    return left * right
                if isinstance(current.op, ast.Div):
                    return left / right
                if isinstance(current.op, ast.Pow):
                    return left ** right
                if isinstance(current.op, ast.Mod):
                    return left % right
            raise ValueError("unsupported")

        try:
            value = evaluate(node)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return f"{value:g}"

    def answer(
        self,
        prompt: str,
        lang: str,
        context: str = "",
        time_info: str = "",
    ) -> str:
        text = (prompt or "").strip()
        if "USER REQUEST:" in text:
            text = text.split("USER REQUEST:", 1)[1].strip()
        lowered = text.lower()
        arabic = lang == "ar"
        if re.search(r"^(stop|quiet|silence|توقف|اسكت|اصمت)$", lowered):
            return "تم الإيقاف." if arabic else "Stopped."
        math_result = self._math(text)
        if math_result is not None:
            return f"{math_result}" if not arabic else f"النتيجة: {math_result}"
        if context.strip():
            header = "أفضل ما وجدته في مصادرك المحلية:" if arabic else "Best local evidence I found:"
            return f"{header}\n\n{context.strip()}\n\n{('هذه مقتطفات من مستنداتك، وليست استنتاجات خارجية.' if arabic else 'These are verbatim passages from your documents, not outside knowledge.')}"
        if re.search(r"^(hello|hi|hey|salam|salaam|مرحبا|أهلا|اهلا|السلام عليكم)", lowered):
            if arabic:
                return (
                    "أهلاً بك. أنا بشرى، مساعدتك المحلية. أعمل دون اتصال وأحتفظ ببياناتك على جهازك. "
                    "يمكنك سؤالي عن الوقت، ملفاتك، مهامك، أو تشغيل مهارة. "
                    "ملاحظة: نموذج Phi-3 غير محمّل حالياً، لذا أستخدم المحرك الاحتياطي المحلي."
                )
            return (
                "Hello. I am Boshra, your local assistant. I work offline and keep your data on this device. "
                "Ask about time, your documents, tasks, device status, or say 'help'. "
                "Note: the Phi-3 model is not loaded, so the deterministic fallback engine is answering."
            )
        if re.search(r"(who are you|your name|من أنت|ما اسمك|عرفي بنفسك)", lowered):
            return (
                "بشرى إصدار 4.0: مساعدة ذكاء اصطناعي محلية السيادة، دون سحابة. المحرك النموذجي المحلي يُحمّل عند الحاجة، والمهارات تُنفّذ عبر أدوات مُتحقَّق منها فقط."
                if arabic else
                f"{APP_NAME} {APP_VERSION}: a sovereign, offline-first assistant. Local models load lazily; skills run only through registered, permission-gated tools. No conversation leaves this device."
            )
        if re.search(r"(help|what can you do|مساعدة|ماذا تستطيع|قدراتك)", lowered):
            return (
                "أستطيع: الوقت والتاريخ، البحث في مستنداتك، تلخيص ملف، إدارة المهام والملاحظات والتذكيرات، تشخيص الجهاز، فتح تطبيق، البحث في الذاكرة، البحث عن الملفات، الأتمتة، الطوارئ دون اتصال، والمزيد عبر المهارات. اطلب 'المهارات' لرؤية القائمة."
                if arabic else
                "I can: tell time/date, search and summarize your vault documents, manage tasks/notes/reminders, run device diagnostics, launch apps, search memory and files, run automation, show offline emergency guidance, and more through skills. Say 'skills' to inspect them."
            )
        if re.search(r"(time|date|الساعة|الوقت|التاريخ|اليوم)", lowered) and time_info:
            return f"{time_info}" if not arabic else time_info
        if re.search(r"(thanks|thank you|شكرا|شكراً)", lowered):
            return "على الرحب والسعة." if arabic else "You are welcome."
        if re.search(r"(model|النموذج|phi)", lowered):
            if arabic:
                return "لم أجد نموذج GGUF محمّلاً. ضع ملف Phi-3 بصيغة GGUF في مجلد النماذج أو استورده من شاشة النماذج، ثم أعد التحميل. حالياً أعمل بالمحرك الاحتياطي المحلي."
            return (
                "No GGUF model is loaded. Import a Phi-3 GGUF file from the Models screen "
                "(or place it in the private models folder), then load it. Until then the "
                "deterministic fallback engine handles requests."
            )
        if arabic:
            return (
                "أعمل حالياً بالمحرك الاحتياطي المحلي، لذلك لا أستطيع توليد نص حر عام. "
                "يمكنني مساعدتك بأدوات محددة: المستندات، الذاكرة، المهام، الجهاز، الطوارئ. "
                "جرّب: «الوقت»، «تشخيص»، «ابحث في ملفاتي عن…»، أو «مساعدة»."
            )
        return (
            "I am running on the deterministic offline fallback, so open-ended generation is unavailable. "
            "I can still execute concrete tools: documents, memory, tasks, device diagnostics, emergency guides. "
            "Try 'time', 'diagnostics', 'search my files for ...', or 'help'."
        )

    def generate_stream(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        stop_event: Optional[threading.Event] = None,
        context: str = "",
        lang: str = "en",
        time_info: str = "",
    ) -> Iterator[str]:
        answer = self.answer(prompt, lang, context=context, time_info=time_info)
        # stream in small slices to preserve the live-terminal feel
        for index in range(0, len(answer), 3):
            if stop_event is not None and stop_event.is_set():
                break
            yield answer[index:index + 3]
            time.sleep(0.004)


class LLMService:
    """Owns exactly one active engine; enforces model lifecycle policy."""

    def __init__(self, config: ConfigManager, models: ModelManager, resources: ResourceManager, store: Store) -> None:
        self.config = config
        self.models = models
        self.resources = resources
        self.store = store
        self._engine: Optional[Any] = None
        self._engine_key = ""
        self._lock = threading.RLock()
        self.fallback = HeuristicEngine()
        self._last_used = 0.0

    @property
    def status(self) -> str:
        with self._lock:
            if self._engine is not None and getattr(self._engine, "available", False):
                return f"{self._engine.name} · {getattr(self._engine, 'status', 'ready')}"
            return "heuristic fallback · no GGUF model loaded"

    @property
    def engine_name(self) -> str:
        with self._lock:
            return self._engine.name if self._engine and getattr(self._engine, "available", False) else self.fallback.name

    def load_selected(self) -> Dict[str, Any]:
        with self._lock:
            model = self.models.selected()
            if model is None:
                return {"ok": False, "error": "No GGUF model found. Import one from Models."}
            if (self._engine is not None and self._engine_key == model.path
                    and getattr(self._engine, "profile", None) == self.resources.profile):
                if self._engine.load():
                    return {"ok": True, "message": f"Model ready: {model.name}"}
            if self._engine is not None:
                self._engine.unload()
            self._engine = LlamaCppEngine(model.path, self.resources.profile)
            self._engine_key = model.path
            if self._engine.load():
                self.store.audit("model", "load", model.name, "low")
                return {"ok": True, "message": self._engine.status}
            error = self._engine.status
            self._engine = None
            self._engine_key = ""
            return {"ok": False, "error": error}

    def unload(self) -> None:
        with self._lock:
            if self._engine is not None:
                name = Path(self._engine.model_path).name
                self._engine.unload()
                self.store.audit("model", "unload", name, "low")
            self._engine = None
            self._engine_key = ""

    def ensure_policy(self) -> None:
        if self.resources.should_unload_model():
            self.unload()
            LOG.warning("model unloaded by resource policy")

    def generate_stream(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        stop_event: Optional[threading.Event],
        context: str = "",
        lang: str = "en",
        time_info: str = "",
    ) -> Iterator[str]:
        with self._lock:
            engine = self._engine
        if engine is not None:
            try:
                yield from engine.generate_stream(
                    prompt, system, max_tokens, temperature, stop_event
                )
                self._last_used = time.time()
                return
            except Exception as exc:  # noqa: BLE001
                LOG.error("LLM engine error, falling back", exc)
                self.store.audit("model", "generation-error", str(exc)[:300], "low")
        yield from self.fallback.generate_stream(
            prompt,
            system,
            max_tokens,
            temperature,
            stop_event,
            context=context,
            lang=lang,
            time_info=time_info,
        )


# --------------------------------------------------------------------------
# Text to speech
# --------------------------------------------------------------------------
class SpeechBackend:
    name = "none"
    available = False
    status = "no speech backend"

    def speak(self, text: str, lang: str, rate: float, stop_event: threading.Event) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        return

    def shutdown(self) -> None:
        return


class AndroidTTSBackend(SpeechBackend):
    name = "android-tts"
    available = False
    status = "not initialised"

    def __init__(self, bridge: AndroidBridge) -> None:
        self.bridge = bridge
        self.engine = None
        self.ready = threading.Event()
        self.init_status = -1
        if not bridge.available:
            self.status = "Android TTS requires Android"
            return
        try:
            from jnius import PythonJavaClass, autoclass, java_method  # type: ignore

            self._autoclass = autoclass
            self._TextToSpeech = autoclass("android.speech.tts.TextToSpeech")
            holder = self

            class _InitListener(PythonJavaClass):
                __javainterfaces__ = ["android/speech/tts/TextToSpeech$OnInitListener"]
                __javacontext__ = "app"

                @java_method("(I)V")
                def onInit(self, status):
                    holder.init_status = int(status)
                    holder.ready.set()

            self._listener = _InitListener()
            self.engine = self._TextToSpeech(bridge.context, self._listener)
            if self.ready.wait(8.0) and self.init_status == 0:
                self.available = True
                self.status = "Android native TextToSpeech ready"
            else:
                self.status = f"TTS init failed (status {self.init_status})"
        except Exception as exc:  # noqa: BLE001
            self.status = f"Android TTS error: {exc}"
            LOG.error("android tts init", exc)

    def speak(self, text: str, lang: str, rate: float, stop_event: threading.Event) -> None:
        if not self.available or self.engine is None:
            return
        try:
            Locale = self._autoclass("java.util.Locale")
            engine = self.engine
            engine.setLanguage(Locale(lang))
            engine.setSpeechRate(float(rate))
            utterance = "boshra-" + uuid.uuid4().hex[:8]
            engine.speak(text, self._TextToSpeech.QUEUE_ADD, None, utterance)
            deadline = time.time() + max(8.0, len(text) / 9.0 + 5.0)
            while time.time() < deadline:
                if stop_event.is_set():
                    engine.stop()
                    return
                try:
                    if not engine.isSpeaking():
                        return
                except Exception:
                    return
                time.sleep(0.15)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"android tts speak failed: {exc}")

    def stop(self) -> None:
        if self.engine is not None:
            with contextlib.suppress(Exception):
                self.engine.stop()

    def shutdown(self) -> None:
        if self.engine is not None:
            with contextlib.suppress(Exception):
                self.engine.stop()
                self.engine.shutdown()


class PiperBackend(SpeechBackend):
    """Piper neural TTS when binary + voice model + audio player are present."""
    name = "piper"
    available = False
    status = "piper assets not found"

    def __init__(self) -> None:
        self.binary = self._find_binary()
        self.player = self._find_player()
        self.voices = self._find_voices()
        self.selected_path = ""
        self.selected_lang = ""
        self._process: Optional[subprocess.Popen] = None
        self._player_process: Optional[subprocess.Popen] = None
        if self.binary and self.voices and self.player:
            self.available = True
            langs = ", ".join(sorted(self.voices))
            self.status = f"Piper ready ({langs}); player {Path(self.player).name}"
        else:
            missing = []
            if not self.binary:
                missing.append("piper binary")
            if not self.voices:
                missing.append("voice .onnx")
            if not self.player:
                missing.append("audio player")
            self.status = "Piper inactive: missing " + ", ".join(missing)

    @staticmethod
    def _find_binary() -> str:
        local = PRIVATE_DIR / "piper" / "piper"
        if local.exists() and os.access(local, os.X_OK):
            return str(local)
        return shutil.which("piper") or ""

    @staticmethod
    def _find_player() -> str:
        for name in ("aplay", "paplay", "ffplay"):
            found = shutil.which(name)
            if found:
                return found
        return ""

    def _find_voices(self) -> Dict[str, str]:
        voices: Dict[str, str] = {}
        for voice in self.list_voices():
            voices.setdefault(voice["lang"], voice["path"])
        return voices

    def set_selected(self, path: str, lang: str = "") -> bool:
        """Remember the user's chosen voice for its language."""
        candidate = str(path or "").strip()
        language = str(lang or "").strip().lower()
        if candidate and language in ("ar", "en") and Path(candidate).exists():
            self.selected_path = candidate
            self.selected_lang = language
            if self.available:
                langs = ", ".join(sorted(self.voices))
                self.status = (f"Piper ready ({langs}); voice {Path(candidate).stem}; "
                               f"player {Path(self.player).name}")
            return True
        self.selected_path = ""
        self.selected_lang = ""
        return False

    def list_voices(self) -> List[Dict[str, Any]]:
        """Every .onnx voice found under the local piper-tts folders."""
        found: List[Dict[str, Any]] = []
        seen: set = set()
        roots = [root / "piper-tts" for root in asset_roots()]
        roots += [PRIVATE_DIR / "piper-tts", Path("/sdcard/assets/piper-tts")]
        for root in roots:
            try:
                if not root.exists():
                    continue
                candidates = sorted(root.rglob("*.onnx"))
            except OSError:
                continue
            for model in candidates:
                key = str(model.resolve())
                if key in seen:
                    continue
                seen.add(key)
                match = re.match(r"^([a-zA-Z]{2})[_-]([a-zA-Z]{2})", model.name)
                lang = (match.group(1).lower() if match else "")
                if lang not in ("ar", "en"):
                    parts = [part.lower() for part in model.parts]
                    lang = "ar" if "ar" in parts or "arabic" in parts else "en"
                try:
                    size = model.stat().st_size
                except OSError:
                    size = 0
                found.append({
                    "name": model.stem,
                    "lang": lang,
                    "path": key,
                    "size": size,
                })
        return sorted(found, key=lambda voice: (voice["lang"], voice["name"].lower()))

    def synthesize(self, text: str, model_path: str, rate: float = 1.0) -> Optional[Path]:
        """Render text with one specific voice. Returns the WAV path or None."""
        if not self.binary or not model_path or not Path(model_path).exists():
            return None
        output = PRIVATE_DIR / "cache" / f"voice_test_{uuid.uuid4().hex[:8]}.wav"
        try:
            length_scale = 1.0 / max(0.5, rate)
            completed = subprocess.run(
                [self.binary, "--model", model_path, "--output_file", str(output),
                 "--length_scale", f"{length_scale:.2f}"],
                input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=max(20.0, len(text) / 6.0),
            )
            if completed.returncode == 0 and output.exists() and output.stat().st_size > 44:
                return output
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"piper voice test failed: {exc}")
        with contextlib.suppress(OSError):
            output.unlink()
        return None

    def speak(self, text: str, lang: str, rate: float, stop_event: threading.Event) -> None:
        model = ""
        if self.selected_path and self.selected_lang == lang and Path(self.selected_path).exists():
            model = self.selected_path
        model = model or self.voices.get(lang) or next(iter(self.voices.values()), "")
        if not model or not self.available:
            return
        output = PRIVATE_DIR / "cache" / f"tts_{uuid.uuid4().hex[:8]}.wav"
        try:
            length_scale = 1.0 / max(0.5, rate)
            process = subprocess.Popen(
                [self.binary, "--model", model, "--output_file", str(output),
                 "--length_scale", f"{length_scale:.2f}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._process = process
            try:
                process.communicate(text.encode("utf-8"), timeout=max(20.0, len(text) / 8.0))
            except subprocess.TimeoutExpired:
                process.kill()
                return
            if stop_event.is_set() or not output.exists():
                return
            player = self.player
            if Path(player).name.startswith("ffplay"):
                command = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(output)]
            else:
                command = [player, str(output)]
            self._player_process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            deadline = time.time() + max(15.0, len(text) / 8.0 + 5.0)
            while self._player_process.poll() is None and time.time() < deadline:
                if stop_event.is_set():
                    with contextlib.suppress(Exception):
                        self._player_process.terminate()
                    break
                time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"piper failed: {exc}")
        finally:
            with contextlib.suppress(OSError):
                output.unlink()

    def stop(self) -> None:
        for process in (self._process, self._player_process):
            if process is not None and process.poll() is None:
                with contextlib.suppress(Exception):
                    process.terminate()

    def shutdown(self) -> None:
        self.stop()


class DesktopTTSBackend(SpeechBackend):
    """speech-dispatcher for desktop verification builds."""
    name = "speech-dispatcher"
    available = False
    status = "spd-say not found"

    def __init__(self) -> None:
        self.binary = shutil.which("spd-say") or ""
        if self.binary:
            self.available = True
            self.status = "speech-dispatcher ready"

    def speak(self, text: str, lang: str, rate: float, stop_event: threading.Event) -> None:
        if not self.available:
            return
        rate_value = int(clamp((rate - 1.0) * 80.0, -80, 80))
        try:
            process = subprocess.Popen(
                [self.binary, "-w", "-l", lang, "-r", str(rate_value), text[:3000]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while process.poll() is None:
                if stop_event.is_set():
                    self.stop()
                    return
                time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"desktop tts failed: {exc}")

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            subprocess.run([self.binary, "-C"], timeout=3,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class SpeechEngine:
    """Single-worker speech queue. No overlapping utterances by construction."""

    SENTENCE_END = re.compile(r"(?<=[.!?؟\n])\s*")

    def __init__(self, bridge: AndroidBridge, config: ConfigManager, on_state: Optional[Callable[[str], None]] = None) -> None:
        self.bridge = bridge
        self.config = config
        self.on_state = on_state
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Tuple[str, str, float]]" = queue.Queue()
        self._buffer = ""
        self._buffer_lang = "en"
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self.piperian = PiperBackend()
        self.piperian.set_selected(
            getattr(config, "piper_voice", ""), getattr(config, "piper_voice_lang", "")
        )
        self.android = AndroidTTSBackend(bridge)
        self.desktop = DesktopTTSBackend() if IS_DESKTOP else None
        self.backend = self._choose_backend()
        self._start_worker()

    def _choose_backend(self) -> SpeechBackend:
        preference = getattr(self.config, "tts_voice", "auto")
        wanted: List[SpeechBackend] = []
        if preference == "piper":
            wanted.append(self.piperian)
        elif preference == "android":
            wanted.append(self.android)
        elif preference == "desktop" and self.desktop:
            wanted.append(self.desktop)
        # "auto" and unknown values begin with the richest backend.
        wanted.extend([self.piperian, self.android, self.desktop])

        ordered: List[SpeechBackend] = []
        for candidate in wanted:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        for candidate in ordered:
            if candidate.available:
                if candidate is not ordered[0]:
                    LOG.info(f"tts: {ordered[0].name} unavailable, using {candidate.name}")
                return candidate
        # even when inactive, keep a backend object so status is truthful
        return ordered[0] if ordered else SpeechBackend()

    @property
    def status(self) -> str:
        return f"{self.backend.name}: {self.backend.status}"

    @property
    def available(self) -> bool:
        return self.backend.available

    def _start_worker(self) -> None:
        def worker() -> None:
            while True:
                try:
                    text, lang, rate = self._queue.get(timeout=0.4)
                except queue.Empty:
                    if self._stop_event.is_set():
                        continue
                    continue
                if text is None:
                    return
                if self._stop_event.is_set():
                    self._queue.task_done()
                    continue
                if self.on_state:
                    with contextlib.suppress(Exception):
                        self.on_state("speaking")
                try:
                    self.backend.speak(text, lang, rate, self._stop_event)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(f"tts backend error: {exc}")
                self._queue.task_done()
                if self._queue.empty() and self.on_state:
                    with contextlib.suppress(Exception):
                        self.on_state("idle")

        self._worker = threading.Thread(target=worker, name="boshra-tts", daemon=True)
        self._worker.start()

    def _split_ready(self, text: str, force: bool = False) -> List[str]:
        if not text:
            return []
        parts = self.SENTENCE_END.split(text)
        if force:
            return [p.strip() for p in parts if p.strip()]
        ready = [p.strip() for p in parts[:-1] if p.strip()]
        self._buffer = parts[-1] if parts else ""
        return ready

    def feed(self, token: str, lang: str) -> None:
        if not self.config.tts_enabled or not self.available:
            return
        with self._lock:
            if lang != self._buffer_lang and self._buffer:
                self._flush_locked()
            self._buffer_lang = lang
            self._buffer += token
            if len(self._buffer) > 1800:
                pieces = self._split_ready(self._buffer, force=True)
                self._buffer = ""
            else:
                pieces = self._split_ready(self._buffer)
            for piece in pieces:
                if len(piece) >= 2:
                    self._queue.put((piece, lang, self.config.tts_rate))

    def _flush_locked(self) -> None:
        if self._buffer.strip():
            self._queue.put((self._buffer.strip(), self._buffer_lang, self.config.tts_rate))
        self._buffer = ""

    def flush(self) -> None:
        if not self.config.tts_enabled or not self.available:
            return
        with self._lock:
            self._flush_locked()

    def speak(self, text: str, lang: Optional[str] = None) -> None:
        if not self.config.tts_enabled or not self.available or not text.strip():
            return
        with self._lock:
            self._flush_locked()
            language = lang or detect_language(text)
            for piece in re.split(r"(?<=[.!?؟\n])\s+", text.strip()):
                if piece.strip():
                    self._queue.put((piece.strip()[:1800], language, self.config.tts_rate))

    def stop(self) -> None:
        with self._lock:
            self._buffer = ""
            while not self._queue.empty():
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()
                    self._queue.task_done()
        self._stop_event.set()
        with contextlib.suppress(Exception):
            self.backend.stop()
        # allow the next utterance again
        threading.Timer(0.2, self._stop_event.clear).start()
        if self.on_state:
            with contextlib.suppress(Exception):
                self.on_state("idle")

    def shutdown(self) -> None:
        self._stop_event.set()
        with contextlib.suppress(Exception):
            self._queue.put((None, "en", 1.0))  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            self.backend.shutdown()


# --------------------------------------------------------------------------
# Voice input (Vosk)
# --------------------------------------------------------------------------
VOSK_CANDIDATE_DIRS = {
    "ar": [
        ASSETS_DIR / "vosk-model-ar",
        PRIVATE_DIR / "vosk-model-ar",
        Path("/sdcard/assets/vosk-model-ar"),
        Path("/sdcard/BoshraAI/models/vosk-model-ar"),
    ],
    "en": [
        ASSETS_DIR / "vosk-model-en",
        PRIVATE_DIR / "vosk-model-en",
        Path("/sdcard/assets/vosk-model-en"),
        Path("/sdcard/BoshraAI/models/vosk-model-en"),
    ],
}


def _valid_vosk_model(path: Path) -> bool:
    return path.is_dir() and ((path / "am").exists() or (path / "conf").exists() or (path / "graph").exists())


def discover_vosk_models() -> Dict[str, str]:
    found: Dict[str, str] = {}
    for lang, candidates in VOSK_CANDIDATE_DIRS.items():
        candidates = [root / f"vosk-model-{lang}" for root in asset_roots()] + list(candidates)
        for candidate in candidates:
            try:
                if _valid_vosk_model(candidate):
                    found[lang] = str(candidate)
                    break
                if candidate.exists() and candidate.is_dir():
                    for child in candidate.iterdir():
                        if child.is_dir() and _valid_vosk_model(child):
                            found[lang] = str(child)
                            break
            except OSError:
                continue
    return found


class AudioSource:
    name = "none"
    available = False
    status = "no audio source"

    def start(self) -> None:
        raise NotImplementedError

    def read(self, frames: int = 3200) -> bytes:
        raise NotImplementedError

    def stop(self) -> None:
        return


class AndroidAudioSource(AudioSource):
    name = "android-audiorecord"

    def __init__(self, bridge: AndroidBridge) -> None:
        self.bridge = bridge
        self.record = None
        self.available = False
        self.status = "not started"
        self._autoclass = None

    def start(self) -> None:
        try:
            from jnius import autoclass  # type: ignore

            self._autoclass = autoclass
            AudioRecord = autoclass("android.media.AudioRecord")
            AudioFormat = autoclass("android.media.AudioFormat")
            MediaRecorder = autoclass("android.media.MediaRecorder$AudioSource")
            self._AudioRecord = AudioRecord
            minimum = int(
                AudioRecord.getMinBufferSize(
                    16000,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                )
            )
            buffer_size = max(minimum, 3200) * 2
            self.record = AudioRecord(
                MediaRecorder.MIC, 16000,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                buffer_size,
            )
            if int(self.record.getState()) != 1:  # STATE_INITIALIZED
                self.status = "AudioRecord not initialised (permission denied?)"
                return
            self.record.startRecording()
            self.available = True
            self.status = "AudioRecord streaming at 16 kHz"
        except Exception as exc:  # noqa: BLE001
            self.status = f"AudioRecord failed: {exc}"
            LOG.error("android audio start", exc)

    def read(self, frames: int = 3200) -> bytes:
        if not self.record:
            return b""
        try:
            from jnius import jarray  # type: ignore

            buffer = jarray("b")([0] * frames)
            count = int(self.record.read(buffer, 0, frames))
            if count <= 0:
                return b""
            return bytes([int(buffer[i]) & 0xFF for i in range(count)])
        except Exception as exc:  # noqa: BLE001
            self.status = f"read failed: {exc}"
            return b""

    def stop(self) -> None:
        if self.record is not None:
            with contextlib.suppress(Exception):
                self.record.stop()
                self.record.release()
            self.record = None
        self.available = False


class DesktopAudioSource(AudioSource):
    name = "sounddevice"

    def __init__(self) -> None:
        self.stream = None
        self.queue: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
        self.available = False
        self.status = "sounddevice not installed"

    def start(self) -> None:
        try:
            import sounddevice as sd  # type: ignore

            def callback(indata, frames, time_info, status):
                with contextlib.suppress(queue.Full):
                    self.queue.put_nowait(bytes(indata))

            self.stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="int16",
                blocksize=1600, callback=callback,
            )
            self.stream.start()
            self.available = True
            self.status = "default microphone streaming"
        except Exception as exc:  # noqa: BLE001
            self.status = f"microphone unavailable: {exc}"

    def read(self, frames: int = 3200) -> bytes:
        try:
            data = self.queue.get(timeout=0.25)
            return data
        except queue.Empty:
            return b""

    def stop(self) -> None:
        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.stop()
                self.stream.close()
            self.stream = None
        self.available = False


class VoiceEngine:
    """Wake-word + command recognition. Foreground only; no hidden recording."""

    def __init__(
        self,
        bridge: AndroidBridge,
        config: ConfigManager,
        resources: ResourceManager,
        store: Store,
        on_state: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.bridge = bridge
        self.config = config
        self.resources = resources
        self.store = store
        self.on_state = on_state
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error
        self.models = discover_vosk_models()
        self._vosk = None
        try:
            import vosk  # type: ignore

            self._vosk = vosk
        except ImportError:
            self._vosk = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._source: Optional[AudioSource] = None
        self.state = "idle"
        self.mode = "command"
        self._locked_lang = "en"

    @property
    def available(self) -> bool:
        return self._vosk is not None and bool(self.models)

    @property
    def status(self) -> str:
        parts = []
        if self._vosk is None:
            parts.append("vosk module missing")
        if not self.models:
            parts.append("no Vosk model folders found")
        if self.models:
            parts.append("models: " + ", ".join(sorted(self.models)))
        return "; ".join(parts) if parts else "ready"

    def _wake_variants(self) -> List[str]:
        configured = normalize_arabic(self.config.wake_word.lower()).strip()
        defaults = [
            "hello boshra", "hey boshra", "hi boshra",
            "مرحبا بشرى", "يا بشرى", "هلا بشرى", configured,
        ]
        return [normalize_arabic(v.lower()) for v in defaults if v]

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"[^\w\u0600-\u06FF ]+", " ", (text or "").lower()).strip()

    def _detect_wake(self, text: str) -> Optional[str]:
        cleaned = self._clean(normalize_arabic(text))
        for variant in self._wake_variants():
            normalized = self._clean(normalize_arabic(variant))
            if normalized and normalized in cleaned:
                remainder = cleaned.replace(normalized, "", 1).strip()
                return remainder
        return None

    def start(self, mode: str = "command") -> bool:
        if not self.available:
            if self.on_error:
                self.on_error(self.status)
            return False
        if self._thread and self._thread.is_alive():
            return True
        if IS_ANDROID and not self.bridge.ensure_permission("RECORD_AUDIO", timeout=45):
            if self.on_error:
                self.on_error("Microphone permission was not granted")
            return False
        self._source = AndroidAudioSource(self.bridge) if IS_ANDROID else DesktopAudioSource()
        self._source.start()
        if not self._source.available:
            if self.on_error:
                self.on_error(self._source.status)
            return False
        self.mode = mode
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="boshra-voice", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._source is not None:
            self._source.stop()
        if self._thread and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.5)
        self._thread = None
        self.state = "idle"
        if self.on_state:
            self.on_state("idle")

    def _choose_langs(self) -> List[str]:
        if self.config.language in ("ar", "en"):
            return [self.config.language] if self.config.language in self.models else list(self.models)[:1]
        if self.config.profile == "LITE":
            return ["en"] if "en" in self.models else list(self.models)[:1]
        return [lang for lang in ("en", "ar") if lang in self.models]

    def _loop(self) -> None:
        assert self._vosk is not None
        models = []
        langs = self._choose_langs()
        try:
            for lang in langs:
                path = self.models[lang]
                model = self._vosk.Model(path)
                recognizer = self._vosk.KaldiRecognizer(model, 16000)
                recognizer.SetWords(False)
                models.append((lang, model, recognizer))
        except Exception as exc:  # noqa: BLE001
            if self.on_error:
                self.on_error(f"Vosk model load failed: {exc}")
            self.stop()
            return
        self.state = "listening"
        if self.on_state:
            self.on_state("listening")
        self.store.audit("voice", "listen-start", f"mode={self.mode} langs={langs}", "low")
        wake_armed = self.mode in ("wake", "continuous")
        last_voice = time.time()
        try:
            while not self._stop.is_set() and self._source is not None:
                data = self._source.read(3200)
                if not data:
                    if time.time() - last_voice > 18:
                        break
                    continue
                last_voice = time.time()
                for lang, _model, recognizer in models:
                    try:
                        if recognizer.AcceptWaveform(data):
                            payload = json.loads(recognizer.Result() or "{}")
                            text = str(payload.get("text", "")).strip()
                            if not text:
                                continue
                            remainder = self._detect_wake(text)
                            if self.mode in ("wake", "continuous") and remainder is not None:
                                if self.on_state:
                                    self.on_state("wake")
                                if remainder and len(remainder) > 2:
                                    self._emit_final(remainder, lang)
                                    if self.mode == "wake":
                                        break
                                else:
                                    wake_armed = True
                                    continue
                            if wake_armed or self.mode == "command":
                                if self.mode == "command" or wake_armed:
                                    self._emit_final(text, lang)
                                    wake_armed = False
                                    if self.mode == "command":
                                        self.stop()
                                        return
                                    if not self.config.continuous_conversation:
                                        self.stop()
                                        return
                        else:
                            partial = json.loads(recognizer.PartialResult() or "{}").get("partial", "")
                            if partial and self.on_partial:
                                self.on_partial(str(partial))
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(f"recognizer error: {exc}")
        finally:
            self.state = "idle"
            if self.on_state:
                self.on_state("idle")
            self.store.audit("voice", "listen-stop", f"mode={self.mode}", "low")
            # release model memory
            models.clear()
            gc.collect()

    def _emit_final(self, text: str, lang: str) -> None:
        if self.on_final:
            self.on_final(text, lang)


# --------------------------------------------------------------------------
# Permissions and risk gate
# --------------------------------------------------------------------------
PERMISSION_CATALOG: Dict[str, Tuple[str, str, str]] = {
    # name: (risk, why, bilingual label)
    "documents.read": ("low", "Read documents you explicitly import or select", "قراءة المستندات"),
    "documents.write": ("medium", "Write summaries and reports into Boshra's folder", "كتابة المستندات"),
    "documents.delete": ("high", "Remove indexed documents from the vault", "حذف المستندات"),
    "files.search": ("low", "Search only approved Boshra folders", "البحث في الملفات"),
    "files.move": ("high", "Move files inside Boshra's workspace", "نقل الملفات"),
    "files.delete": ("high", "Delete files inside Boshra's workspace", "حذف الملفات"),
    "camera.capture": ("medium", "Capture a photo when you request it", "الكاميرا"),
    "microphone.listen": ("medium", "Listen only while the microphone control is active", "الميكروفون"),
    "notifications.read": ("high", "Read notifications (requires a native listener service)", "قراءة الإشعارات"),
    "notifications.show": ("low", "Show local reminders and alerts", "إظهار التنبيهات"),
    "network.access": ("medium", "Use the internet for skills you invoke", "الوصول للشبكة"),
    "android.app.open": ("medium", "Open another installed application", "فتح التطبيقات"),
    "calendar.read": ("low", "Read local calendar entries stored by Boshra", "قراءة التقويم"),
    "calendar.write": ("medium", "Create local calendar entries", "كتابة التقويم"),
    "clipboard.read": ("high", "Read the system clipboard when asked", "قراءة الحافظة"),
    "clipboard.write": ("low", "Copy text to the clipboard", "كتابة الحافظة"),
    "memory.read": ("low", "Retrieve what you asked Boshra to remember", "قراءة الذاكرة"),
    "memory.write": ("low", "Store preferences and notes you approve", "كتابة الذاكرة"),
    "device.info": ("low", "Read local device, battery and storage telemetry", "معلومات الجهاز"),
    "tts.speak": ("low", "Speak responses aloud", "النطق"),
    "skills.execute": ("medium", "Run locally installed skills", "تنفيذ المهارات"),
}


class PermissionManager:
    def __init__(self, store: Store, config: ConfigManager) -> None:
        self.store = store
        self.config = config
        self.confirm_handler: Optional[Callable[[str, str], bool]] = None
        self._lock = threading.RLock()

    def has(self, permission: str) -> bool:
        row = self.store.query_one("SELECT granted FROM perms WHERE permission=?", (permission,))
        return bool(row and row["granted"])

    def grant(self, permission: str, reason: str = "") -> bool:
        if permission not in PERMISSION_CATALOG:
            return False
        self.store.execute(
            "INSERT INTO perms(permission, granted, reason, ts) VALUES(?,1,?,?) "
            "ON CONFLICT(permission) DO UPDATE SET granted=1, reason=excluded.reason, ts=excluded.ts",
            (permission, reason[:400], now_iso()),
        )
        self.store.audit("permission", "grant", permission, PERMISSION_CATALOG[permission][0])
        return True

    def revoke(self, permission: str) -> bool:
        self.store.execute(
            "INSERT INTO perms(permission, granted, reason, ts) VALUES(?,0,'',?) "
            "ON CONFLICT(permission) DO UPDATE SET granted=0, ts=excluded.ts",
            (permission, now_iso()),
        )
        self.store.audit("permission", "revoke", permission, "medium")
        return True

    def grants(self) -> Dict[str, bool]:
        rows = self.store.query("SELECT permission, granted FROM perms")
        return {row["permission"]: bool(row["granted"]) for row in rows}

    def request(self, permission: str, detail: str = "") -> bool:
        if permission not in PERMISSION_CATALOG:
            return False
        if self.has(permission):
            return True
        risk, why, label = PERMISSION_CATALOG[permission]
        question = (
            f"Allow {label} ({permission})?\n\nReason: {why}\nRisk: {risk.upper()}"
            + (f"\n\n{detail}" if detail else "")
        )
        if self.confirm_handler is not None:
            accepted = bool(self.confirm_handler(question, detail))
        else:
            LOG.warning(f"permission prompt with no UI handler: {permission}")
            accepted = False
        if accepted:
            self.grant(permission, why)
        return accepted

    def require(
        self,
        permissions: Sequence[str],
        purpose: str,
        risk: str = "low",
        detail: str = "",
        force_confirm: bool = False,
    ) -> Tuple[bool, str]:
        for permission in permissions:
            if permission not in PERMISSION_CATALOG:
                return False, f"Unknown permission requested: {permission}"
            if not self.request(permission, purpose):
                return False, f"Permission denied: {permission}"
        worst = max(
            [risk] + [PERMISSION_CATALOG[p][0] for p in permissions],
            key=lambda r: {"low": 0, "medium": 1, "high": 2}.get(r, 0),
        )
        if (worst == "high" and self.config.confirm_high_risk) or force_confirm:
            question = f"HIGH-RISK ACTION\n\n{purpose}\n\n{detail}\n\nProceed?"
            if self.confirm_handler is None:
                return False, "High-risk action requires explicit confirmation (no UI available)"
            if not self.confirm_handler(question, detail):
                return False, "Action cancelled by user"
        return True, "authorized"


# --------------------------------------------------------------------------
# Application context and tools
# --------------------------------------------------------------------------
@dataclass
class ToolResult:
    ok: bool
    message: str = ""
    data: Any = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    spoken: str = ""
    needs_confirmation: bool = False
    preview: str = ""
    error: Optional[str] = None
    navigate: str = ""

    def render(self) -> str:
        parts = [self.message or (self.preview if self.needs_confirmation else "")]
        if self.sources:
            formatted = ", ".join(
                f"{s.get('title', '?')}" + (f" p.{s['page']}" if s.get("page") else "")
                for s in self.sources[:6]
            )
            parts.append(f"\nSources: {formatted}")
        return "\n".join(p for p in parts if p).strip() or ("Done" if self.ok else (self.error or "Failed"))


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable[["AppContext", Dict[str, Any]], ToolResult]
    permissions: Tuple[str, ...] = ()
    risk: str = "low"
    destructive: bool = False
    needs_network: bool = False
    preview: Optional[Callable[["AppContext", Dict[str, Any]], ToolResult]] = None


class AppContext:
    """Service locator shared by tools, skills and the orchestrator."""

    def __init__(self) -> None:
        self.store: Store = None  # type: ignore[assignment]
        self.config: ConfigManager = None  # type: ignore[assignment]
        self.memory: MemoryManager = None  # type: ignore[assignment]
        self.vault: KnowledgeVault = None  # type: ignore[assignment]
        self.bridge: AndroidBridge = None  # type: ignore[assignment]
        self.resources: ResourceManager = None  # type: ignore[assignment]
        self.models: ModelManager = None  # type: ignore[assignment]
        self.llm: LLMService = None  # type: ignore[assignment]
        self.tts: SpeechEngine = None  # type: ignore[assignment]
        self.voice: Optional[VoiceEngine] = None
        self.permissions: PermissionManager = None  # type: ignore[assignment]
        self.skills: Any = None
        self.automation: Any = None
        self.recall: Any = None
        self.backups: Any = None
        self.planner: Any = None
        self.proactive: Any = None
        self.compactor: Any = None
        self.orchestrator: Any = None
        self.tools: ToolRegistry = None  # type: ignore[assignment]
        self.diagnostics: Any = None
        self.emergency: Any = None
        self.online: Any = None
        self.confirm_handler: Optional[Callable[[str, str], bool]] = None
        self.ui_emit: Callable[[str, Dict[str, Any]], None] = lambda event, payload: None
        self.pending: Optional[Dict[str, Any]] = None
        self.focus_end: float = 0.0
        self.recording_steps: List[Dict[str, Any]] = []
        self._event_lock = threading.RLock()

    # -- UI notifications ------------------------------------------------
    def emit(self, event: str, **payload: Any) -> None:
        try:
            self.ui_emit(event, payload)
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"ui emit failed {event}: {exc}")

    def confirm(self, question: str, detail: str = "") -> bool:
        if self.confirm_handler is None:
            LOG.warning("confirmation required but no UI handler is attached")
            return False
        try:
            return bool(self.confirm_handler(question, detail))
        except Exception as exc:  # noqa: BLE001
            LOG.error("confirm handler failed", exc)
            return False

    def set_pending(self, tool: str, args: Dict[str, Any], preview: str) -> None:
        with self._event_lock:
            self.pending = {"tool": tool, "args": dict(args), "preview": preview}

    def take_pending(self) -> Optional[Dict[str, Any]]:
        with self._event_lock:
            pending = self.pending
            self.pending = None
            return pending

    def record_step(self, tool: str, args: Dict[str, Any]) -> None:
        if self.config.recording:
            with self._event_lock:
                self.recording_steps.append({"tool": tool, "args": args})


class ToolRegistry:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self) -> List[ToolSpec]:
        return [self._tools[name] for name in self.names()]

    def execute(self, name: str, args: Optional[Dict[str, Any]] = None) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(False, error=f"Unknown tool: {name}")
        args = dict(args or {})
        try:
            if spec.needs_network:
                if self.ctx.config.online_mode == "local":
                    return ToolResult(
                        False,
                        error="Network access is disabled (Online mode: LOCAL ONLY).",
                    )
                network = self.ctx.bridge.network_info()
                if not network.get("online"):
                    return ToolResult(False, error="No internet connection is available.")
            authorized, reason = self.ctx.permissions.require(
                spec.permissions,
                purpose=f"{spec.name}: {spec.description}",
                risk=spec.risk,
                detail=json.dumps(args, ensure_ascii=False)[:500],
            )
            if not authorized:
                return ToolResult(False, error=reason)
            if spec.destructive and not args.get("_confirmed"):
                if spec.preview is None:
                    return ToolResult(False, error="Destructive tool lacks a preview implementation")
                preview = spec.preview(self.ctx, args)
                if not preview.ok:
                    return preview
                preview.needs_confirmation = True
                preview.preview = preview.preview or preview.message
                self.ctx.set_pending(name, args, preview.message)
                return preview
            clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
            result = spec.handler(self.ctx, clean_args)
            self.ctx.record_step(name, clean_args)
            self.ctx.store.audit(
                "tool",
                name,
                (result.message or result.error or "")[:1000],
                spec.risk,
                online=spec.needs_network,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"tool {name} crashed", exc)
            self.ctx.store.audit("tool", f"{name}-error", str(exc)[:1000], "medium")
            return ToolResult(False, error=f"Tool failed: {exc}")


# -- helpers ----------------------------------------------------------------
def _text(args: Dict[str, Any], key: str, required: bool = True) -> str:
    value = args.get(key, "")
    value = str(value).strip() if value is not None else ""
    if required and not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def _int(args: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(args.get(key, default))
    except (TypeError, ValueError):
        return default


def allowed_roots(ctx: AppContext) -> List[Path]:
    roots = [WORKSPACE_DIR, EXPORT_DIR, PRIVATE_DIR / "vault"]
    for folder in ctx.config.vault_folders:
        path = Path(folder)
        if path.exists() and path.is_dir():
            roots.append(path)
    return roots


def _resolve_allowed(path_value: str, ctx: AppContext) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = WORKSPACE_DIR / path
    if not path.exists():
        raise ValueError("File does not exist")
    for root in allowed_roots(ctx):
        if is_within(path, root):
            return path
    if is_within(path, PRIVATE_DIR / "vault"):
        return path
    raise ValueError("Path is outside the approved Boshra folders")


def _iter_files(roots: Sequence[Path], max_files: int = 20000, max_depth: int = 10) -> Iterator[Path]:
    seen = 0
    skip = {"__pycache__", ".git", ".thumbnails", "node_modules", "Android", ".Trash"}
    for root in roots:
        if not root.exists():
            continue
        stack: List[Tuple[Path, int]] = [(root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                if entry.name.startswith(".") or entry.name in skip:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if depth < max_depth:
                            stack.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        yield Path(entry.path)
                        seen += 1
                        if seen >= max_files:
                            return
                except OSError:
                    continue


def _find_document(ctx: AppContext, name_or_id: str) -> Optional[Dict[str, Any]]:
    if not name_or_id:
        return None
    if name_or_id.isdigit():
        return ctx.store.query_one("SELECT * FROM documents WHERE id=?", (int(name_or_id),))
    rows = ctx.vault.documents()
    needle = normalize_arabic(name_or_id.lower())
    for row in rows:
        if needle in normalize_arabic(str(row["title"]).lower()):
            return row
    return rows[0] if rows else None


# -- tool implementations ---------------------------------------------------
def tool_time(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    now = datetime.now()
    lang = args.get("lang") or ("ar" if contains_arabic(args.get("query", "")) else "en")
    days_ar = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"]
    months_ar = ["جانفي", "فيفري", "مارس", "أفريل", "ماي", "جوان", "جويلية",
                 "أوت", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
    if lang == "ar":
        text = f"{days_ar[now.weekday()]} {now.day} {months_ar[now.month - 1]} {now.year} — {now.strftime('%H:%M')}"
    else:
        text = now.strftime("%A, %d %B %Y — %H:%M")
    return ToolResult(True, text, data={"iso": now.isoformat(), "epoch": now.timestamp()}, spoken=text)


def tool_device_info(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    info = ctx.bridge.device_info()
    memory = ctx.bridge.memory_info()
    device = info.get("model", "?")
    maker = info.get("manufacturer", "?")
    android = info.get("android", "")
    sdk = info.get("sdk", "")
    lines = [
        f"Device: {maker} {device}",
        (f"Android {android} (API {sdk})" if android else f"Platform: {info.get('platform')}"),
        f"ABI/CPU: {info.get('cpu', '?')} · {info.get('cores', '?')} cores",
        f"RAM: {human_bytes(memory.get('available', 0))} free / {human_bytes(memory.get('total', 0))}",
    ]
    text = "\n".join(lines)
    return ToolResult(True, text, data=info, spoken=text)


def tool_battery(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    info = ctx.bridge.battery_info()
    if info.get("percent") is None:
        return ToolResult(False, error="Battery telemetry is unavailable on this device/build.")
    charging = "charging" if info.get("charging") else "not charging"
    temp = info.get("temperature_c")
    text = f"Battery {info['percent']}% · {charging}" + (f" · {temp}°C" if temp else "")
    return ToolResult(True, text, data=info, spoken=text)


def tool_storage(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    info = ctx.bridge.storage_info()
    if not info.get("total"):
        return ToolResult(False, error="Storage telemetry unavailable")
    text = f"Storage: {human_bytes(info['free'])} free of {human_bytes(info['total'])}"
    return ToolResult(True, text, data=info, spoken=text)


def tool_open_app(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    query = _text(args, "app")
    result = ctx.bridge.open_app(query)
    if result.get("ok"):
        return ToolResult(True, result.get("message", "Opened"), data=result, spoken=result.get("message", ""))
    return ToolResult(False, error=result.get("error", "Could not open app"))


def tool_file_search(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    query = _text(args, "query", required=False)
    suffix = str(args.get("extension", "")).lower().lstrip(".")
    limit = max(1, min(_int(args, "limit", 25), 100))
    if not query and not suffix:
        raise ValueError("Provide a search term or extension")
    matches: List[Dict[str, Any]] = []
    needle = normalize_arabic(query.lower())
    for path in _iter_files(allowed_roots(ctx)):
        try:
            name_norm = normalize_arabic(path.name.lower())
            if needle and needle not in name_norm and needle not in normalize_arabic(str(path.parent).lower()):
                continue
            if suffix and path.suffix.lower() != "." + suffix:
                continue
            stat = path.stat()
            matches.append(
                {"name": path.name, "path": str(path), "size": stat.st_size, "modified": stat.st_mtime}
            )
        except OSError:
            continue
        if len(matches) >= limit * 4:
            break
    matches.sort(key=lambda m: m["modified"], reverse=True)
    matches = matches[:limit]
    if not matches:
        return ToolResult(True, "No matching files in approved Boshra folders.", data={"files": []})
    lines = [f"{m['name']}  ({human_bytes(m['size'])})" for m in matches]
    text = "Files found:\n" + "\n".join(lines)
    return ToolResult(True, text, data={"files": matches}, spoken=f"Found {len(matches)} files.")


def tool_read_document(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    path = _resolve_allowed(_text(args, "path"), ctx)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, pages = extract_pdf(path)
    elif suffix == ".docx":
        text, pages = extract_docx(path)
    elif suffix in IMAGE_EXTENSIONS:
        text, pages = ocr_image_file(path), 1
    elif suffix in TEXT_EXTENSIONS:
        text, pages = extract_text_file(path), 0
    else:
        return ToolResult(False, error=f"Unsupported format: {suffix}")
    excerpt = text.strip()[:9000]
    return ToolResult(
        True,
        excerpt or "No text found",
        data={"path": str(path), "pages": pages, "chars": len(text)},
        sources=[{"title": path.name, "page": 0}],
        spoken=(excerpt[:400] + "…") if excerpt else "",
    )


def tool_vault_import(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    path = _text(args, "path")
    source = Path(path)
    if not source.exists():
        found = tool_file_search(ctx, {"query": source.name, "limit": 5})
        files = (found.data or {}).get("files", [])
        if not files:
            return ToolResult(False, error=f"File not found: {path}")
        source = Path(files[0]["path"])
    allowed = any(is_within(source, root) for root in allowed_roots(ctx))
    if not allowed:
        return ToolResult(False, error="Only files inside approved Boshra folders can be imported")
    result = ctx.vault.import_file(source, title=str(args.get("title", "")))
    if not result.get("ok"):
        return ToolResult(False, error=result.get("error", "Import failed"))
    return ToolResult(True, result.get("message", "Imported"), data=result)


def tool_vault_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    docs = ctx.vault.documents()
    if not docs:
        return ToolResult(True, "The Knowledge Vault is empty.", data={"documents": []})
    lines = [
        f"[{d['id']}] {d['title']} · {d['pages'] or '?'} pages · {human_ts(d['added'])}"
        for d in docs
    ]
    return ToolResult(True, "Vault documents:\n" + "\n".join(lines), data={"documents": docs})


def tool_vault_search(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    query = _text(args, "query")
    results = ctx.vault.search(query, k=_int(args, "k", 5))
    if not results:
        return ToolResult(True, "No matching passages found.", data={"results": []})
    lines = [
        f"[{r['title']}{' p.' + str(r['page']) if r['page'] else ''}] {r['text'][:420]}…"
        for r in results
    ]
    sources = [{"title": r["title"], "page": r["page"], "doc_id": r["doc_id"]} for r in results]
    return ToolResult(True, "Relevant passages:\n\n" + "\n\n".join(lines), data={"results": results}, sources=sources)


def tool_vault_answer(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    question = _text(args, "query")
    result = ctx.vault.answer(question, k=_int(args, "k", 4))
    if not result.get("ok"):
        return ToolResult(False, error=result.get("error", "No answer"))
    text = result["answer"]
    return ToolResult(True, text, data=result, sources=result["sources"], spoken=text[:600])


def tool_vault_summarize(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    doc = _find_document(ctx, str(args.get("document", "")))
    if not doc:
        return ToolResult(False, error="No document found. Import a document first.")
    result = ctx.vault.summarize(int(doc["id"]), max_sentences=_int(args, "sentences", 6))
    if not result.get("ok"):
        return ToolResult(False, error=result.get("error", "Summary failed"))
    text = f"Summary of {doc['title']}:\n{result['summary']}"
    return ToolResult(True, text, data=result, sources=[{"title": doc["title"], "page": 0}], spoken=text[:800])


def tool_vault_export(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    doc = _find_document(ctx, str(args.get("document", "")))
    if not doc:
        return ToolResult(False, error="No document found")
    result = ctx.vault.export_summary(int(doc["id"]))
    if not result.get("ok"):
        return ToolResult(False, error=result.get("error", "Export failed"))
    return ToolResult(True, f"Summary exported to {result['path']}", data=result)


def tool_vault_delete(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    doc = _find_document(ctx, str(args.get("document", "")))
    if not doc:
        return ToolResult(False, error="Document not found")
    ctx.vault.delete_document(int(doc["id"]))
    return ToolResult(True, f"Deleted {doc['title']} from the vault.")


def tool_memory_add(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    content = _text(args, "content")
    result = ctx.memory.add(
        str(args.get("category", "semantic")),
        content,
        key=str(args.get("key", "")),
        source="user",
        tags=str(args.get("tags", "")),
    )
    if not result.get("ok"):
        return ToolResult(False, error=result.get("error", "Memory refused"))
    return ToolResult(True, "Remembered locally.", data=result)


def tool_memory_search(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    query = _text(args, "query")
    results = ctx.memory.search(query, k=_int(args, "k", 6))
    if not results:
        return ToolResult(True, "Nothing relevant in memory.", data={"results": []})
    lines = [
        f"- [{r['category']}] {r['content'][:500]} (source: {r['source']}, {human_ts(r['updated'])})"
        for r in results
    ]
    return ToolResult(True, "From your local memory:\n" + "\n".join(lines), data={"results": results})


def tool_memory_delete(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    memory_id = _int(args, "id", 0)
    if not memory_id:
        raise ValueError("Memory id required")
    ctx.memory.delete(memory_id)
    return ToolResult(True, f"Deleted memory #{memory_id}")


def tool_daily_briefing(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    """A deterministic agenda from local records; never invents priorities."""
    lang = str(args.get("lang", "en"))
    ar = lang == "ar"
    now = datetime.now()
    today = now.date().isoformat()
    tomorrow = (now.date() + timedelta(days=1)).isoformat()
    tasks = ctx.store.query(
        "SELECT id, title, due FROM tasks WHERE done=0 "
        "ORDER BY CASE WHEN due='' THEN 1 ELSE 0 END, due, id LIMIT 8"
    )
    count = ctx.store.query_one("SELECT COUNT(*) AS n FROM tasks WHERE done=0")["n"]
    reminders = ctx.store.query(
        "SELECT title, at FROM reminders WHERE fired=0 AND at<? ORDER BY at LIMIT 8", (tomorrow,)
    )
    events = ctx.store.query(
        "SELECT title, start FROM events WHERE start>=? AND start<? ORDER BY start LIMIT 8",
        (today, tomorrow),
    )
    lines = [("موجز اليوم · " if ar else "Daily briefing · ") + today,
             (f"المهام المفتوحة: {count}" if ar else f"Open tasks: {count}")]
    if tasks:
        lines.append("\nالخطوات التالية" if ar else "\nNext actions")
        for task in tasks:
            due = str(task["due"] or "")
            overdue = bool(due and due[:10] < today)
            suffix = (" · متأخرة" if ar else " · overdue") if overdue else ""
            lines.append(f"○ [{task['id']}] {task['title']}" + (f" · {due}" if due else "") + suffix)
    lines.append("\nتذكيرات حتى نهاية اليوم" if ar else "\nReminders through today")
    lines.extend(f"• {r['title']} · {human_ts(r['at'])}" for r in reminders)
    if not reminders:
        lines.append("لا توجد تذكيرات معلقة اليوم." if ar else "No pending reminders today.")
    lines.append("\nتقويم اليوم" if ar else "\nToday's calendar")
    lines.extend(f"• {e['title']} · {human_ts(e['start'])}" for e in events)
    if not events:
        lines.append("لا توجد أحداث اليوم." if ar else "No events today.")
    remaining = max(0, int((float(ctx.focus_end or 0) - time.time()) / 60))
    if remaining:
        lines.append(f"\nالتركيز: {remaining} دقيقة متبقية" if ar else f"\nFocus: {remaining} minutes remaining")
    lines.append("\nالمصدر: السجلات المحلية. حتى 8 عناصر لكل قسم." if ar else
                 "\nSource: local records. Up to 8 items per section.")
    return ToolResult(True, "\n".join(lines), data={
        "tasks": tasks, "open_tasks": count, "reminders": reminders, "events": events,
    })


def tool_task_add(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    title = _text(args, "title")
    due = str(args.get("due", ""))[:40]
    task_id = ctx.store.execute(
        "INSERT INTO tasks(title, done, due, created) VALUES(?,0,?,?)",
        (title[:300], due, now_iso()),
    )
    return ToolResult(True, f"Task added: {title}", data={"id": task_id})


def tool_task_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    rows = ctx.store.query("SELECT * FROM tasks ORDER BY done ASC, id DESC LIMIT 100")
    if not rows:
        return ToolResult(True, "No tasks yet.", data={"tasks": []})
    lines = [f"{'✓' if row['done'] else '○'} [{row['id']}] {row['title']}" for row in rows]
    return ToolResult(True, "Tasks:\n" + "\n".join(lines), data={"tasks": rows})


def tool_task_done(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    task_id = _int(args, "id", 0)
    if not task_id:
        raise ValueError("Task id required")
    if not ctx.store.query_one("SELECT id FROM tasks WHERE id=?", (task_id,)):
        return ToolResult(False, error=f"Task #{task_id} does not exist")
    ctx.store.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
    return ToolResult(True, f"Task #{task_id} completed")


def tool_note_add(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    title = _text(args, "title")
    body = str(args.get("body", ""))
    sealed, enc = ctx.store.seal(body)
    note_id = ctx.store.execute(
        "INSERT INTO notes(title, body, enc, created) VALUES(?,?,?,?)",
        (title[:200], sealed, enc, now_iso()),
    )
    return ToolResult(True, f"Note saved: {title}", data={"id": note_id})


def tool_note_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    rows = ctx.store.query("SELECT * FROM notes ORDER BY id DESC LIMIT 50")
    if not rows:
        return ToolResult(True, "No notes yet.", data={"notes": []})
    lines = [f"[{row['id']}] {row['title']} — {human_ts(row['created'])}" for row in rows]
    return ToolResult(True, "Notes:\n" + "\n".join(lines), data={"notes": rows})


def tool_reminder_add(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    title = _text(args, "title")
    when = str(args.get("at", "")).strip()
    if not when:
        minutes = _int(args, "minutes", 0)
        if minutes <= 0:
            raise ValueError("Provide 'at' ISO time or 'minutes'")
        when = (datetime.now() + timedelta(minutes=minutes)).isoformat(timespec="seconds")
    parsed = parse_iso(when)
    if not parsed:
        return ToolResult(False, error="Invalid time format; use YYYY-MM-DDTHH:MM")
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    reminder_id = ctx.store.execute(
        "INSERT INTO reminders(title, at, fired, created) VALUES(?,?,0,?)",
        (title[:200], parsed.isoformat(timespec="seconds"), now_iso()),
    )
    return ToolResult(True, f"Reminder set for {parsed.strftime('%Y-%m-%d %H:%M')}", data={"id": reminder_id})


def tool_reminder_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    rows = ctx.store.query("SELECT * FROM reminders ORDER BY at LIMIT 100")
    if not rows:
        return ToolResult(True, "No reminders.", data={"reminders": []})
    lines = [f"[{row['id']}] {row['title']} — {human_ts(row['at'])} {'✓' if row['fired'] else ''}" for row in rows]
    return ToolResult(True, "Reminders:\n" + "\n".join(lines), data={"reminders": rows})


def tool_event_add(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    title = _text(args, "title")
    start = str(args.get("start", "")).strip() or now_iso()
    end = str(args.get("end", "")).strip()
    event_id = ctx.store.execute(
        "INSERT INTO events(title, start, end, notes, created) VALUES(?,?,?,?,?)",
        (title[:200], start, end, str(args.get("notes", ""))[:2000], now_iso()),
    )
    return ToolResult(True, f"Local calendar entry added: {title}", data={"id": event_id})


def tool_event_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    rows = ctx.store.query("SELECT * FROM events ORDER BY start LIMIT 100")
    if not rows:
        return ToolResult(True, "No calendar entries.", data={"events": []})
    lines = [f"- {row['title']} · {row['start']}" for row in rows]
    return ToolResult(True, "Local calendar:\n" + "\n".join(lines), data={"events": rows})


def tool_focus_timer(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    minutes = max(1, min(_int(args, "minutes", 25), 240))
    ctx.focus_end = time.time() + minutes * 60
    ctx.store.state_set("focus_end", str(ctx.focus_end))
    ctx.emit("focus", minutes=minutes)
    return ToolResult(True, f"Focus session started for {minutes} minutes.", data={"end": ctx.focus_end})


def tool_focus_stop(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    ctx.focus_end = 0.0
    ctx.store.state_set("focus_end", "0")
    return ToolResult(True, "Focus session stopped.")


def tool_emergency(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    query = str(args.get("query", args.get("topic", "")))
    if ctx.emergency is None:
        return ToolResult(False, error="Emergency module unavailable")
    result = ctx.emergency.search(query)
    return ToolResult(
        True,
        result["text"],
        data=result,
        sources=[{"title": "Boshra offline emergency guide", "page": 0}],
        spoken=result.get("spoken", ""),
    )


# -- online tools ------------------------------------------------------------
def _http_get(url: str, timeout: float = 12.0, headers: Optional[Dict[str, str]] = None) -> str:
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError("Only http(s) URLs are allowed")
    host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0].lower()
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise ValueError("Local network targets are blocked")
    import requests  # type: ignore

    default_headers = {"User-Agent": f"{APP_NAME}/{APP_VERSION} (offline assistant; user-invoked)"}
    if headers:
        default_headers.update(headers)
    response = requests.get(url, headers=default_headers, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def tool_web_search(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    query = _text(args, "query")
    url = "https://html.duckduckgo.com/html/?q=" + __import__("urllib.parse").parse.quote(query)
    try:
        page = _http_get(url)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, error=f"Web search failed (offline or blocked): {exc}")
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
    )
    results = []
    for match in pattern.finditer(page):
        href = html.unescape(match.group(1))
        title = re.sub(r"<[^>]+>", "", html.unescape(match.group(2))).strip()
        if "uddg=" in href:
            parsed = __import__("urllib.parse").parse.parse_qs(
                __import__("urllib.parse").parse.urlparse(href).query
            )
            href = parsed.get("uddg", [href])[0]
        if title and href.startswith("http"):
            results.append({"title": title, "url": href})
        if len(results) >= 6:
            break
    if not results:
        return ToolResult(False, error="No results parsed; the search provider may have changed.")
    lines = [f"{i + 1}. {r['title']}\n   {r['url']}" for i, r in enumerate(results)]
    text = f"Web results for '{query}' (external content):\n" + "\n".join(lines)
    return ToolResult(True, text, data={"results": results}, spoken=f"Found {len(results)} web results.")


def _parse_rss(xml_text: str, limit: int = 8) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_text.encode("utf-8", "ignore"))
    items = root.findall(".//item")
    if not items:
        items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
    results: List[Dict[str, str]] = []
    for item in items[:limit]:
        title = (item.findtext("title") or item.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not link:
            link_node = item.find("{http://www.w3.org/2005/Atom}link")
            link = link_node.get("href", "") if link_node is not None else ""
        published = (item.findtext("pubDate") or item.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        if title:
            results.append({"title": html.unescape(re.sub(r"<[^>]+>", "", title)), "link": link, "published": published})
    return results


def tool_news(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    feeds = [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("الجزيرة", "https://www.aljazeera.net/aljazeerarss/a7c186be-1baa-4bd4-be85-8c3f3c9c92de/9a5b6a5c-2f7b-4fbf-9a5e-6f0f0f0f0f0f"),
    ]
    all_items: List[Dict[str, str]] = []
    for name, url in feeds:
        try:
            for item in _parse_rss(_http_get(url), limit=5):
                item["feed"] = name
                all_items.append(item)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"news feed failed {name}: {exc}")
    if not all_items:
        return ToolResult(False, error="News is unavailable (offline or feeds blocked).")
    lines = [f"- [{i['feed']}] {i['title']}" for i in all_items[:10]]
    return ToolResult(
        True,
        "Headlines (external content):\n" + "\n".join(lines),
        data={"items": all_items},
        spoken="Headlines are ready on screen.",
        sources=[{"title": i["feed"], "page": 0} for i in all_items[:3]],
    )


def tool_weather(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    city = _text(args, "city")
    try:
        import requests  # type: ignore

        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload["current_condition"][0]
        text = (
            f"Weather in {city}: {current['weatherDesc'][0]['value']}, "
            f"{current['temp_C']}°C (feels {current['FeelsLikeC']}°C), "
            f"humidity {current['humidity']}%, wind {current['windspeedKmph']} km/h."
        )
        return ToolResult(
            True, text,
            data={"city": city, "current": current},
            sources=[{"title": "wttr.in", "page": 0}],
            spoken=text,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, error=f"Weather lookup failed: {exc}")


def tool_rss(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    url = _text(args, "url")
    try:
        items = _parse_rss(_http_get(url), limit=_int(args, "limit", 8))
    except Exception as exc:  # noqa: BLE001
        return ToolResult(False, error=f"RSS fetch failed: {exc}")
    if not items:
        return ToolResult(False, error="No RSS items found")
    lines = [f"- {item['title']}" for item in items]
    return ToolResult(True, "RSS items:\n" + "\n".join(lines), data={"items": items})


# -- translation (offline dictionary + optional local LLM) -------------------
OFFLINE_DICTIONARY = {
    "hello": "مرحبا", "hi": "مرحبا", "good morning": "صباح الخير", "good evening": "مساء الخير",
    "thank you": "شكرا", "thanks": "شكرا", "please": "من فضلك", "yes": "نعم", "no": "لا",
    "water": "ماء", "food": "طعام", "help": "مساعدة", "doctor": "طبيب", "hospital": "مستشفى",
    "police": "شرطة", "fire": "حريق", "emergency": "طوارئ", "danger": "خطر", "safe": "آمن",
    "home": "منزل", "car": "سيارة", "road": "طريق", "work": "عمل", "school": "مدرسة",
    "today": "اليوم", "tomorrow": "غدا", "yesterday": "أمس", "time": "وقت", "day": "يوم",
    "night": "ليل", "morning": "صباح", "evening": "مساء", "friend": "صديق", "family": "عائلة",
    "love": "حب", "peace": "سلام", "computer": "حاسوب", "phone": "هاتف", "battery": "بطارية",
    "document": "مستند", "file": "ملف", "memory": "ذاكرة", "search": "بحث", "open": "افتح",
    "close": "أغلق", "stop": "توقف", "start": "ابدأ", "read": "اقرأ", "write": "اكتب",
    "translate": "ترجم", "language": "لغة", "English": "الإنجليزية", "Arabic": "العربية",
    "ماء": "water", "طعام": "food", "مساعدة": "help", "طبيب": "doctor", "مستشفى": "hospital",
    "شرطة": "police", "حريق": "fire", "طوارئ": "emergency", "خطر": "danger", "آمن": "safe",
    "منزل": "home", "سيارة": "car", "طريق": "road", "عمل": "work", "مدرسة": "school",
    "اليوم": "today", "غدا": "tomorrow", "أمس": "yesterday", "وقت": "time", "يوم": "day",
    "ليل": "night", "صباح": "morning", "مساء": "evening", "صديق": "friend", "عائلة": "family",
    "حب": "love", "سلام": "peace", "حاسوب": "computer", "هاتف": "phone", "بطارية": "battery",
    "مستند": "document", "ملف": "file", "ذاكرة": "memory", "بحث": "search", "شكرا": "thank you",
    "مرحبا": "hello", "نعم": "yes", "لا": "no",
}


def _offline_translate(text: str, target: str) -> str:
    source_lang = detect_language(text)
    if source_lang == target:
        return text
    words = re.findall(r"[\w\u0600-\u06FF]+", text)
    translated = []
    missing = 0
    for word in words:
        key = word.lower()
        if key in OFFLINE_DICTIONARY:
            translated.append(OFFLINE_DICTIONARY[key])
        else:
            translated.append(word)
            missing += 1
    result = " ".join(translated)
    if missing > len(words) * 0.4:
        result += (
            "\n\n(Offline dictionary only; load a local GGUF model for fluent translation.)"
            if target == "en" else
            "\n\n(قاموس محلي فقط؛ حمّل نموذج GGUF محلي للترجمة السلسة.)"
        )
    return result


def tool_translate(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    text = _text(args, "text")
    target = str(args.get("target", "")).lower()
    if target not in ("ar", "en"):
        target = "en" if detect_language(text) == "ar" else "ar"
    engine_available = False
    try:
        if ctx.llm.engine_name.startswith("llama"):
            prompt = f"Translate the following text into {'Arabic' if target == 'ar' else 'English'}. Output only the translation:\n{text}"
            chunks = list(ctx.llm.generate_stream(prompt, "You are a precise translator.", 300, 0.2, None, lang=target))
            translation = "".join(chunks).strip()
            engine_available = bool(translation)
        else:
            translation = _offline_translate(text, target)
    except Exception:  # noqa: BLE001
        translation = _offline_translate(text, target)
    return ToolResult(True, translation, data={"target": target, "model": engine_available})


# -- code tools --------------------------------------------------------------
def _parse_code(code: str) -> Any:
    import ast

    return ast.parse(code)


def tool_code_explain(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    code = _text(args, "code")
    try:
        tree = _parse_code(code)
    except SyntaxError as exc:
        return ToolResult(False, error=f"Syntax error: {exc}")
    import ast

    lines: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            lines.append("Imports: " + ", ".join(alias.name for alias in node.names))
        elif isinstance(node, ast.ImportFrom):
            lines.append(f"From {node.module}: " + ", ".join(a.name for a in node.names))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args_list = [a.arg for a in node.args.args]
            doc = ast.get_docstring(node) or ""
            lines.append(f"Function {node.name}({', '.join(args_list)}) — {doc[:120]}")
            lines.append(f"  branches: {sum(1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.For, ast.While, ast.Try)))}")
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            lines.append(f"Class {node.name}: {', '.join(methods) or 'no methods'}")
        elif isinstance(node, ast.Assign):
            targets = ", ".join(t.id for t in node.targets if isinstance(t, ast.Name))
            if targets:
                lines.append(f"Assignment: {targets}")
    if not lines:
        return ToolResult(True, "The snippet contains only simple expressions.")
    return ToolResult(True, "Static analysis:\n" + "\n".join(lines))


SECURITY_PATTERNS = [
    (r"\beval\s*\(", "HIGH", "eval() enables arbitrary code execution"),
    (r"\bexec\s*\(", "HIGH", "exec() enables arbitrary code execution"),
    (r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True", "HIGH", "shell=True risks command injection"),
    (r"pickle\.loads?\s*\(", "HIGH", "Unpickling untrusted data is unsafe"),
    (r"verify\s*=\s*False", "HIGH", "TLS verification disabled"),
    (r"hashlib\.md5\s*\(", "MEDIUM", "MD5 is not suitable for security"),
    (r"except\s*:\s*(?:\n|$)", "MEDIUM", "Bare except hides errors"),
    (r"os\.system\s*\(", "HIGH", "os.system risks command injection"),
    (r"yaml\.load\s*\([^)]*\)", "MEDIUM", "Use yaml.safe_load for untrusted YAML"),
    (r"SELECT .*\+.*(?:where|WHERE)", "MEDIUM", "Possible SQL concatenation"),
]


def tool_secure_review(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    code = _text(args, "code")
    findings: List[str] = []
    for pattern, severity, description in SECURITY_PATTERNS:
        for match in re.finditer(pattern, code, re.IGNORECASE):
            line = code[: match.start()].count("\n") + 1
            findings.append(f"[{severity}] line {line}: {description}")
    if not findings:
        return ToolResult(True, "No known insecure patterns matched. This is not a full security audit.")
    return ToolResult(True, "Security review findings:\n" + "\n".join(findings), data={"findings": findings})


def tool_refactor(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    code = _text(args, "code")
    try:
        tree = _parse_code(code)
    except SyntaxError as exc:
        return ToolResult(False, error=f"Syntax error: {exc}")
    import ast

    suggestions: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
            if length > 40:
                suggestions.append(f"{node.name}: {length} lines — consider splitting responsibilities.")
            if len(node.args.args) > 6:
                suggestions.append(f"{node.name}: {len(node.args.args)} parameters — consider a config object.")
            defaults = node.args.defaults
            for default in defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    suggestions.append(f"{node.name}: mutable default argument — use None and initialise inside.")
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            suggestions.append(f"line {node.lineno}: bare except — catch specific exceptions.")
    if not suggestions:
        return ToolResult(True, "No structural refactoring hotspots found by the local heuristics.")
    return ToolResult(True, "Refactoring suggestions:\n" + "\n".join(f"- {s}" for s in suggestions))


def tool_test_generate(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    code = _text(args, "code")
    try:
        tree = _parse_code(code)
    except SyntaxError as exc:
        return ToolResult(False, error=f"Syntax error: {exc}")
    import ast

    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not functions:
        return ToolResult(False, error="No top-level functions found to test.")
    module = str(args.get("module", "module"))
    lines = ["import pytest", f"from {module} import " + ", ".join(f.name for f in functions), ""]
    for function in functions:
        args_list = [a.arg for a in function.args.args]
        lines.append(f"def test_{function.name}_smoke():")
        lines.append(f"    # arrange concrete inputs for {', '.join(args_list) or 'no args'}")
        call = f"{function.name}(" + ", ".join("None" for _ in args_list) + ")"
        lines.append(f"    result = {call}")
        lines.append("    assert result is not None  # replace with a meaningful assertion")
        lines.append("")
    return ToolResult(True, "Generated pytest skeleton (review before use):\n```python\n" + "\n".join(lines) + "```")


def tool_debug(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    traceback_text = _text(args, "traceback")
    exception_match = re.findall(r"^([A-Za-z_][\w.]*Error|[A-Za-z_][\w.]*Exception)\s*:?\s*(.*)$", traceback_text, re.MULTILINE)
    location_match = re.findall(r'File "([^"]+)", line (\d+)', traceback_text)
    hints: List[str] = []
    if location_match:
        file_name, line_no = location_match[-1]
        hints.append(f"Last frame: {file_name}:{line_no}")
    if exception_match:
        kind, message = exception_match[-1]
        hints.append(f"Exception: {kind} — {message or 'no message'}")
        if "KeyError" in kind:
            hints.append("Check dictionary keys or use .get() with a default.")
        elif "TypeError" in kind:
            hints.append("Verify argument types and optional None values.")
        elif "AttributeError" in kind:
            hints.append("The object may be None or an unexpected type; log it before use.")
        elif "FileNotFoundError" in kind:
            hints.append("Resolve paths against app-private storage, not the shell CWD.")
        elif "PermissionError" in kind:
            hints.append("Android storage permissions or scoped storage may be blocking this path.")
    else:
        hints.append("No Python exception header recognised; include the full traceback text.")
    return ToolResult(True, "Debug analysis:\n" + "\n".join(f"- {h}" for h in hints))


def tool_report(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    title = str(args.get("title", "Boshra report"))[:80]
    body = str(args.get("body", ""))
    if not body:
        return ToolResult(False, error="Report body is empty")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:60] or "report"
    path = EXPORT_DIR / f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    content = f"# {title}\n\n_Generated by {APP_NAME} {APP_VERSION} on {now_iso()}_\n\n{body}\n"
    if not atomic_write_text(path, content, 0o644):
        return ToolResult(False, error="Could not write report")
    return ToolResult(True, f"Report written to {path}", data={"path": str(path)})


def tool_organize_preview(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    folder_value = str(args.get("folder", "")).strip()
    base = Path(folder_value) if folder_value else WORKSPACE_DIR / "inbox"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
    if not is_within(base, WORKSPACE_DIR):
        return ToolResult(False, error="Organizer only works inside BoshraAI/workspace")
    categories = {
        "Documents": {".pdf", ".docx", ".txt", ".md", ".csv"},
        "Images": {".png", ".jpg", ".jpeg", ".webp"},
        "Audio": {".mp3", ".wav", ".m4a", ".ogg"},
        "Video": {".mp4", ".mkv", ".webm"},
        "Archives": {".zip", ".tar", ".gz", ".7z"},
        "Code": {".py", ".js", ".kt", ".java", ".c", ".cpp", ".json", ".xml", ".html"},
    }
    moves: List[Tuple[str, str]] = []
    duplicates: List[str] = []
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        target_category = "Other"
        for category, extensions in categories.items():
            if path.suffix.lower() in extensions:
                target_category = category
                break
        target = base / target_category / path.name
        if target.exists():
            duplicates.append(str(path))
            moves.append((str(path), str(base / "_duplicates" / path.name)))
        else:
            moves.append((str(path), str(target)))
    lines = [f"Move: {Path(src).name} → {Path(dst).parent.name}/" for src, dst in moves[:40]]
    if duplicates:
        lines.append(f"Duplicates moved to _duplicates/: {len(duplicates)}")
    preview = f"Shadow preview for {base} ({len(moves)} files; no changes made):\n" + "\n".join(lines)
    return ToolResult(True, preview, data={"moves": moves, "folder": str(base)}, preview=preview)


def tool_organize_apply(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    preview = tool_organize_preview(ctx, args)
    if not preview.ok:
        return preview
    moves = (preview.data or {}).get("moves", [])
    done = 0
    failures: List[str] = []
    for source, destination in moves:
        try:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
            done += 1
        except OSError as exc:
            failures.append(f"{Path(source).name}: {exc}")
    ctx.store.audit("files", "organize", f"{done} moved, {len(failures)} failed", "high")
    message = f"Organized {done} files."
    if failures:
        message += "\nFailures:\n" + "\n".join(failures[:10])
    return ToolResult(True, message)


def tool_clipboard_read(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    text = ctx.bridge.clipboard_get()
    if not text:
        return ToolResult(False, error="Clipboard is empty or access was denied by Android")
    ctx.store.audit("clipboard", "read", f"{len(text)} chars", "high")
    return ToolResult(True, text[:3000])


def tool_clipboard_write(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    text = _text(args, "text")
    if not ctx.bridge.clipboard_set(text):
        return ToolResult(False, error="Clipboard write unavailable")
    return ToolResult(True, "Copied to clipboard.")


def tool_notify(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    title = str(args.get("title", APP_NAME))
    message = _text(args, "message")
    if ctx.bridge.notify(title, message):
        return ToolResult(True, "Notification shown.")
    return ToolResult(False, error="Notifications are unavailable on this build.")


def tool_theme(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    theme = _text(args, "theme").lower()
    if theme not in VALID_THEMES:
        return ToolResult(False, error=f"Unknown theme. Choose: {', '.join(VALID_THEMES)}")
    ctx.config.set("theme", theme)
    ctx.emit("theme", theme=theme)
    return ToolResult(True, f"Theme set to {theme}.")


def tool_profile(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    profile = _text(args, "profile").upper()
    if profile not in VALID_PROFILES:
        return ToolResult(False, error=f"Choose one of: {', '.join(VALID_PROFILES)}")
    ctx.config.set("profile", profile)
    ctx.emit("profile", profile=profile)
    return ToolResult(True, f"Hardware profile set to {profile}.")


def tool_skill_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.skills is None:
        return ToolResult(False, error="Skill engine unavailable")
    skills = ctx.skills.list_skills()
    lines = [
        f"{'●' if s['enabled'] else '○'} {s['name']} ({s['id']}) · {s['category']} · {s['risk']}"
        for s in skills
    ]
    return ToolResult(True, "Installed skills:\n" + "\n".join(lines), data={"skills": skills})


def tool_skill_execute(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.skills is None:
        return ToolResult(False, error="Skill engine unavailable")
    skill_id = _text(args, "skill")
    command = str(args.get("command", "")).strip()
    data = args.get("data") if isinstance(args.get("data"), dict) else {}
    result = ctx.skills.execute(skill_id, command, data)
    return ToolResult(result.success, result.message, data=result.to_dict(), error=result.error)


def tool_workflow_run(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.automation is None:
        return ToolResult(False, error="Automation engine unavailable")
    name = _text(args, "name")
    result = ctx.automation.run_by_name(name)
    return ToolResult(result.get("ok", False), result.get("message", ""), data=result, error=result.get("error"))


def tool_math(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    expression = _text(args, "expression")
    answer = HeuristicEngine()._math(expression)
    if answer is None:
        return ToolResult(False, error="Could not evaluate that expression")
    return ToolResult(True, answer, spoken=answer)


def tool_recall(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.recall is None:
        return ToolResult(False, error="Recall index unavailable")
    query = _text(args, "query")
    kinds = args.get("kinds")
    ctx.recall.reindex_if_stale()
    if kinds and isinstance(kinds, str):
        kinds = [part.strip() for part in kinds.split(",") if part.strip()]
    hits = ctx.recall.search(query, k=int(args.get("k") or 10),
                             kinds=kinds if isinstance(kinds, list) else None)
    if not hits:
        return ToolResult(True, T(f"No local matches for “{query}”.",
                                  f"لا توجد نتائج محلية عن «{query}»."), data={"hits": []})
    lines = [T(f"Recall for “{query}”", f"استرجاع عن «{query}»")]
    for hit in hits[:8]:
        label = RECALL_KINDS.get(hit["kind"], (hit["kind"], hit["kind"]))
        lines.append(f"• [{T(*label)} · {hit['title']}] {hit['snippet']}")
    return ToolResult(
        True, "\n".join(lines), data={"hits": hits[:8]},
        sources=[{"title": hit["title"], "kind": hit["kind"], "score": hit["score"]}
                 for hit in hits[:8]],
    )


def tool_recall_reindex(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.recall is None:
        return ToolResult(False, error="Recall index unavailable")
    counts = ctx.recall.reindex()
    total = sum(counts.values())
    detail = ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))
    return ToolResult(True, T(f"Recall index rebuilt · {total} items ({detail})",
                              f"أُعيد بناء فهرس الاسترجاع · {total} عنصر ({detail})"),
                      data=counts)


def tool_plan_create(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.planner is None:
        return ToolResult(False, error="Plan engine unavailable")
    goal = _text(args, "goal")
    outcome = ctx.planner.propose(goal)
    if not outcome.get("ok"):
        return ToolResult(False, error=outcome.get("error") or "Could not build the plan")
    plan = outcome["plan"]
    lines = [T(f"Plan #{plan['id']}: {plan['goal']}", f"خطة #{plan['id']}: {plan['goal']}")]
    for index, step in enumerate(plan.get("steps") or [], 1):
        target = step.get("tool") or step.get("routine") or step.get("kind")
        lines.append(f"{index}. {step['title']} → {target}")
    lines.append(T("Say “run plan” to execute it step by step.",
                   "قل «نفّذ الخطة» لتنفيذها خطوة بخطوة."))
    return ToolResult(True, "\n".join(lines), data={"plan": plan})


def tool_plan_run(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.planner is None:
        return ToolResult(False, error="Plan engine unavailable")
    plan_id = int(args.get("plan_id") or 0)
    if not plan_id:
        recent = ctx.planner.recent(1)
        if not recent:
            return ToolResult(False, error="There is no plan to run")
        plan_id = int(recent[0]["id"])
    outcome = ctx.planner.run(plan_id)
    if not outcome.get("ok"):
        return ToolResult(
            False, outcome.get("summary", ""), data=outcome,
            error=outcome.get("error") or T("Plan stopped before completion",
                                            "توقفت الخطة قبل الاكتمال"),
        )
    return ToolResult(True, outcome.get("summary", ""), data=outcome)


def tool_plan_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.planner is None:
        return ToolResult(False, error="Plan engine unavailable")
    plans = ctx.planner.recent(int(args.get("limit") or 6))
    if not plans:
        return ToolResult(True, T("No plans yet.", "لا توجد خطط بعد."))
    lines = [T("Recent plans", "خطط حديثة")]
    for plan in plans:
        steps = plan.get("steps") or []
        done = sum(1 for step in steps if step.get("status") == "done")
        lines.append(f"#{plan['id']} · {plan['status']} · {done}/{len(steps)} · {plan['goal'][:90]}")
    return ToolResult(True, "\n".join(lines), data={"plans": plans})


def tool_backup_create(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.backups is None:
        return ToolResult(False, error="Backup manager unavailable")
    password = str(args.get("password") or "")
    label = str(args.get("label") or "manual")
    outcome = ctx.backups.create(password=password, label=label)
    if not outcome.get("ok"):
        return ToolResult(False, error=outcome.get("error") or "Backup failed")
    return ToolResult(True, outcome.get("message", ""), data=outcome)


def tool_backup_list(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.backups is None:
        return ToolResult(False, error="Backup manager unavailable")
    entries = ctx.backups.list_backups()
    if not entries:
        return ToolResult(True, T("No backups yet.", "لا توجد نسخ احتياطية بعد."))
    lines = [T("Encrypted backups", "نسخ احتياطية مشفرة")]
    for entry in entries[:10]:
        size = int(entry.get("size") or 0) / 1024
        lines.append(f"• {Path(str(entry.get('path', ''))).name} · {entry.get('mode')} · "
                     f"{size:.0f} KB · {str(entry.get('created', ''))[:16]}")
    return ToolResult(True, "\n".join(lines), data={"backups": entries})


def tool_backup_restore(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.backups is None:
        return ToolResult(False, error="Backup manager unavailable")
    path = str(args.get("path") or "")
    if not path:
        known = ctx.backups.list_backups()
        if not known:
            return ToolResult(False, error="There is no backup to restore")
        path = str(known[0].get("path", ""))
    outcome = ctx.backups.restore(path, password=str(args.get("password") or ""))
    if not outcome.get("ok"):
        return ToolResult(False, error=outcome.get("error") or "Restore failed")
    return ToolResult(True, outcome.get("message", ""), data=outcome)


def tool_digest(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.proactive is None:
        return ToolResult(False, error="Proactive engine unavailable")
    kind = str(args.get("kind") or "morning").lower()
    forced = kind == "manual"
    digest = ctx.proactive.check(force=forced) if forced else ctx.proactive.build(kind)
    if forced and digest is None:
        digest = ctx.proactive.build("manual")
    return ToolResult(True, str(digest.get("body", "")), data=digest)


def tool_commitments(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.proactive is None:
        return ToolResult(False, error="Proactive engine unavailable")
    action = str(args.get("action") or "list").lower()
    if action in ("done", "dismiss", "dismissed") and args.get("id"):
        ctx.proactive.resolve_commitment(int(args["id"]), "dismissed" if action.startswith("dismiss") else "done")
        return ToolResult(True, T("Follow-up updated.", "تم تحديث المتابعة."))
    items = ctx.proactive.open_commitments(10)
    if not items:
        return ToolResult(True, T("No open follow-ups.", "لا توجد متابعات مفتوحة."))
    lines = [T("Open follow-ups", "متابعات مفتوحة")]
    for item in items:
        lines.append(f"#{item['id']} · {item['text']}")
    return ToolResult(True, "\n".join(lines), data={"commitments": items})


def tool_compact(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    if ctx.compactor is None:
        return ToolResult(False, error="Compaction engine unavailable")
    outcome = ctx.compactor.compact(force=bool(args.get("force")))
    if not outcome.get("ok"):
        return ToolResult(False, error=outcome.get("error") or "Nothing to compact")
    usage = ctx.compactor.usage()
    return ToolResult(
        True,
        T(f"Conversation compacted · {outcome['count']} earlier turns summarized\n"
          f"Context budget: {usage['used']}/{usage['limit']} tokens ({usage['percent']}%)",
          f"ضُغطت المحادثة · تلخيص {outcome['count']} دوراً سابقاً\n"
          f"ميزانية السياق: {usage['used']}/{usage['limit']} رمز ({usage['percent']}%)"),
        data={**outcome, "usage": usage},
    )


def tool_float(ctx: AppContext, args: Dict[str, Any]) -> ToolResult:
    action = str(args.get("action") or "status").lower()
    bridge = ctx.bridge
    if action == "start":
        outcome = bridge.float_start(int(args.get("size") or 58),
                                     ctx.config.sphere_color, ctx.config.sphere_style)
        if not outcome.get("ok"):
            return ToolResult(False, error=outcome.get("error") or "Could not start the float")
        ctx.config.set("float_enabled", True)
        return ToolResult(True, T("Floating assistant is on. It stays above other apps.",
                                  "المساعد العائم يعمل الآن فوق التطبيقات الأخرى."),
                          data=outcome)
    if action == "stop":
        outcome = bridge.float_stop()
        ctx.config.set("float_enabled", False)
        if not outcome.get("ok"):
            return ToolResult(False, error=outcome.get("error") or "Could not stop the float")
        return ToolResult(True, T("Floating assistant hidden.", "أُخفي المساعد العائم."),
                          data=outcome)
    if action == "permission":
        ok = bridge.request_overlay_permission()
        return ToolResult(ok, T("Open Android settings and allow “display over other apps”.",
                                "افتح إعدادات أندرويد واسمح بـ«العرض فوق التطبيقات الأخرى».")
                          if ok else "",
                          error=None if ok else "Overlay settings are unavailable")
    status = bridge.float_status()
    status["enabled"] = bool(ctx.config.float_enabled)
    if status.get("running") and status.get("suppressed"):
        status["running"] = False
    label = T("running", "يعمل") if status["running"] else T("stopped", "متوقف")
    lines = [
        T(f"Floating assistant · {label}", f"المساعد العائم · {label}"),
        T(f"supported: {'yes' if status['supported'] else 'no'} · "
          f"overlay permission: {'granted' if status['granted'] else 'missing'}",
          f"مدعوم: {'نعم' if status['supported'] else 'لا'} · "
          f"إذن العرض: {'ممنوح' if status['granted'] else 'مفقود'}"),
    ]
    return ToolResult(bool(status["supported"]), "\n".join(lines), data=status)


# --------------------------------------------------------------------------
# Offline emergency & survival guide
# --------------------------------------------------------------------------
class EmergencyGuide:
    """Bundled offline guidance. Informational only, not medical authority."""

    TOPICS: Dict[str, Dict[str, Any]] = {
        "cpr": {
            "title_en": "CPR (adult)", "title_ar": "الإنعاش القلبي الرئوي (بالغ)",
            "keywords": ["cpr", "heart", "breathing", "unconscious", "إنعاش", "قلب", "تنفس", "فاقد الوعي"],
            "steps_en": [
                "Call emergency services immediately; put the phone on speaker.",
                "Check responsiveness and normal breathing for no more than 10 seconds.",
                "If unresponsive and not breathing normally, start chest compressions: centre of chest, 5–6 cm deep, 100–120 per minute.",
                "Give 30 compressions then 2 rescue breaths if trained and willing; otherwise continue hands-only compressions.",
                "Use an AED as soon as one is available and follow its voice prompts.",
                "Continue until emergency services take over, the person recovers, or you are exhausted.",
            ],
            "steps_ar": [
                "اتصل بالطوارئ فوراً وفعّل مكبر الصوت.",
                "تحقق من الاستجابة والتنفس الطبيعي لمدة لا تتجاوز 10 ثوانٍ.",
                "إذا كان فاقد الوعي ولا يتنفس طبيعياً، ابدأ ضغطات الصدر: منتصف الصدر، عمق 5–6 سم، بمعدل 100–120 في الدقيقة.",
                "30 ضغطة ثم نفسان إن كنت مدرباً ومستعداً؛ وإلا فواصل الضغطات فقط.",
                "استخدم جهاز الصدمات (AED) فور توفره واتبع تعليماته الصوتية.",
                "واصل حتى وصول الطوارئ أو تعافي المصاب أو إجهادك.",
            ],
        },
        "choking": {
            "title_en": "Choking (adult)", "title_ar": "الاختناق (بالغ)",
            "keywords": ["choke", "choking", "airway", "اختناق", "غصة", "مجاري"],
            "steps_en": [
                "If the person can cough forcefully, encourage coughing and stay with them.",
                "If they cannot breathe, speak or cough, call emergency services.",
                "Give up to 5 sharp back blows between the shoulder blades.",
                "If ineffective, give up to 5 abdominal thrusts (Heimlich manoeuvre).",
                "Alternate 5 back blows and 5 thrusts until the object clears or help arrives.",
                "If they become unresponsive, begin CPR and tell emergency services.",
            ],
            "steps_ar": [
                "إذا كان قادراً على السعال بقوة، شجعه على السعال وابق معه.",
                "إذا لم يستطع التنفس أو الكلام أو السعال، اتصل بالطوارئ.",
                "وجّه حتى 5 ضربات حادة على الظهر بين الكتفين.",
                "إن لم تنجح، نفّذ حتى 5 ضغطات بطنية (مناورة هايمليك).",
                "بادل بين 5 ضربات ظهر و5 ضغطات حتى خروج الجسم أو وصول المساعدة.",
                "إذا فقد الوعي، ابدأ الإنعاش وأخبر الطوارئ.",
            ],
        },
        "bleeding": {
            "title_en": "Severe bleeding", "title_ar": "نزيف حاد",
            "keywords": ["bleed", "blood", "wound", "نزيف", "دم", "جرح"],
            "steps_en": [
                "Call emergency services.",
                "Apply firm, continuous pressure directly on the wound with a clean cloth.",
                "Do not remove a soaked dressing; add more layers on top.",
                "Elevate the limb if possible and no fracture is suspected.",
                "Use a tourniquet only for life-threatening limb bleeding when pressure fails; note the time.",
                "Keep the person warm and monitor breathing until help arrives.",
            ],
            "steps_ar": [
                "اتصل بالطوارئ.",
                "اضغط بقوة ومستمرة مباشرة على الجرح بقطعة قماش نظيفة.",
                "لا تُزل الضمادة المتشبعة بالدم؛ أضف طبقات فوقها.",
                "ارفع الطرف إن أمكن ولا يُشتبه بكسر.",
                "استخدم الرباط الضاغط فقط لنزيف طرف مهدد للحياة عند فشل الضغط؛ وسجّل الوقت.",
                "أبقِ المصاب دافئاً وراقب تنفسه حتى وصول المساعدة.",
            ],
        },
        "fire": {
            "title_en": "Fire", "title_ar": "حريق",
            "keywords": ["fire", "smoke", "burn", "حريق", "دخان", "حرق"],
            "steps_en": [
                "Raise the alarm and call emergency services from a safe place.",
                "Evacuate immediately; do not collect belongings.",
                "Stay low under smoke and close doors behind you to slow the fire.",
                "Never use elevators. Use stairs.",
                "If clothing catches fire: stop, drop and roll; cool burns with running water.",
                "If trapped, seal gaps under doors, cover vents, signal from a window and call for help.",
            ],
            "steps_ar": [
                "أطلق الإنذار واتصل بالطوارئ من مكان آمن.",
                "أخلِ المكان فوراً؛ لا تجمع الممتلكات.",
                "ابق منخفضاً تحت الدخان وأغلق الأبواب خلفك لإبطاء الحريق.",
                "لا تستخدم المصاعد أبداً؛ استخدم السلالم.",
                "إذا اشتعلت الملابس: توقف واسقط وتدحرج؛ وبرّد الحروق بماء جارٍ.",
                "إذا احتُجزت: أغلق الفجوات أسفل الأبواب، وغطِّ الفتحات، وأعطِ إشارات من النافذة واطلب المساعدة.",
            ],
        },
        "earthquake": {
            "title_en": "Earthquake", "title_ar": "زلزال",
            "keywords": ["earthquake", "quake", "زلزال", "هزة"],
            "steps_en": [
                "Drop to your hands and knees, cover your head and neck, hold on to sturdy furniture.",
                "Stay away from windows, shelves and exterior walls.",
                "Do not run outside during shaking.",
                "After shaking stops, evacuate carefully, avoiding damaged structures and power lines.",
                "Expect aftershocks; keep shoes and a light near you.",
                "Use text messages rather than calls to reduce network congestion.",
            ],
            "steps_ar": [
                "انزل على يديك وركبتيك، واحمِ رأسك ورقبتك، وتشبث بأثاث متين.",
                "ابتعد عن النوافذ والرفوف والجدران الخارجية.",
                "لا تخرج أثناء الاهتزاز.",
                "بعد توقف الهزة، أخلِ المكان بحذر مع تجنب المباني المتضررة وخطوط الكهرباء.",
                "توقع هزات ارتدادية؛ وأبقِ حذاءً وضوءاً قريباً.",
                "استخدم الرسائل النصية بدل الاتصالات لتخفيف الضغط على الشبكة.",
            ],
        },
        "flood": {
            "title_en": "Flood", "title_ar": "فيضان",
            "keywords": ["flood", "water", "فيضان", "مياه", "سيل"],
            "steps_en": [
                "Move to higher ground immediately; avoid walking or driving through moving water.",
                "Stay away from bridges, riverbanks and power lines.",
                "Turn off electricity only if it is safe to reach the breaker.",
                "Do not drink flood water; treat water before use.",
                "Signal for help with a light or bright cloth if trapped.",
            ],
            "steps_ar": [
                "انتقل إلى أرض مرتفعة فوراً؛ تجنب المشي أو القيادة عبر المياه المتدفقة.",
                "ابتعد عن الجسور والضفاف وخطوط الكهرباء.",
                "اقطع الكهرباء فقط إذا كان الوصول إلى القاطع آمناً.",
                "لا تشرب مياه الفيضان؛ عالج المياه قبل الاستخدام.",
                "أعطِ إشارات للمساعدة بضوء أو قماش واضح إن احتُجزت.",
            ],
        },
        "power": {
            "title_en": "Power outage", "title_ar": "انقطاع الكهرباء",
            "keywords": ["power", "outage", "electricity", "كهرباء", "انقطاع"],
            "steps_en": [
                "Use a torch rather than candles to prevent fire.",
                "Keep the refrigerator closed; food stays cold for several hours.",
                "Unplug sensitive electronics to protect them from surges.",
                "Preserve phone battery: lower brightness, enable power saving, share updates by text.",
                "Keep medicines that need refrigeration in a cooler with ice.",
                "Never run a generator indoors or near windows.",
            ],
            "steps_ar": [
                "استخدم مصباحاً بدل الشموع لمنع الحرائق.",
                "أبقِ الثلاجة مغلقة؛ يبقى الطعام بارداً لعدة ساعات.",
                "افصل الأجهزة الحساسة لحمايتها من تقلبات الجهد.",
                "حافظ على بطارية الهاتف: خفّض السطوع، فعّل توفير الطاقة، وأرسل تحديثات نصية.",
                "ضع الأدوية التي تحتاج تبريداً في مبرد مع ثلج.",
                "لا تشغّل المولد في مكان مغلق أو قرب النوافذ.",
            ],
        },
        "survival": {
            "title_en": "Survival basics", "title_ar": "أساسيات البقاء",
            "keywords": ["survival", "lost", "shelter", "بقاء", "تائه", "مأوى", "نجاة"],
            "steps_en": [
                "Stop, stay calm, and think before acting. Tell someone your plan if possible.",
                "Prefer shelter and warmth first; exposure kills faster than hunger.",
                "Water is the first priority; do not drink untreated water if you can boil, filter or disinfect it.",
                "Signal for rescue: three of anything (whistles, lights, fires) is a recognised distress pattern.",
                "If rescue is expected, stay near your last known location and make yourself visible.",
                "Do not eat unknown plants or mushrooms.",
            ],
            "steps_ar": [
                "توقف، واهدأ، وفكّر قبل التصرف. أخبر أحداً بخطتك إن أمكن.",
                "أعطِ الأولوية للمأوى والدفء؛ التعرض للبرد أخطر من الجوع.",
                "الماء أولاً؛ لا تشرب ماءً غير معالج إن أمكن غليه أو ترشيحه أو تطهيره.",
                "أشارات الإنقاذ: ثلاثة من أي شيء (صافرات، أضواء، نيران) نمط معترف به للاستغاثة.",
                "إذا كان الإنقاذ متوقعاً، ابقَ قرب آخر موقع معروف واجعل نفسك مرئياً.",
                "لا تأكل نباتات أو فطوراً مجهولة.",
            ],
        },
    }

    NUMBERS = [
        ("International (GSM / EU)", "112"),
        ("Algeria · Protection Civile", "14"),
        ("Algeria · Police", "17"),
        ("Algeria · Gendarmerie", "1055"),
        ("France · SAMU / Police / Fire", "15 / 17 / 18"),
        ("United Kingdom", "999 or 112"),
        ("United States / Canada", "911"),
        ("Tunisia · Police / Ambulance", "197 / 190"),
        ("Morocco · Police / Ambulance", "19 / 15"),
        ("Saudi Arabia", "911"),
        ("UAE", "999 / 998"),
        ("Deaf / hard of hearing (EU)", "112 app or SMS where available"),
    ]

    def __init__(self, config: ConfigManager) -> None:
        self.config = config

    def _lang(self, query: str) -> str:
        if self.config.language in ("ar", "en"):
            return self.config.language
        return "ar" if contains_arabic(query) else "en"

    def numbers_text(self, lang: str) -> str:
        header = "أرقام الطوارئ (تحقق محلياً؛ قد تتغير):" if lang == "ar" else "Emergency numbers (verify locally; they can change):"
        return header + "\n" + "\n".join(f"• {name}: {number}" for name, number in self.NUMBERS)

    def search(self, query: str) -> Dict[str, Any]:
        lang = self._lang(query)
        needle = normalize_arabic((query or "").lower())
        if not needle or needle in ("overview", "نظرة عامة") or re.search(r"(number|رقم|أرقام|numbers|المساعدة|help)", needle):
            if not needle:
                topics = list(self.TOPICS)
                body = (
                    "دليل الطوارئ دون اتصال. اذكر: حريق، اختناق، نزيف، إنعاش، زلزال، فيضان، انقطاع كهرباء، بقاء."
                    if lang == "ar" else
                    "Offline emergency guide. Say: fire, choking, bleeding, CPR, earthquake, flood, power outage, survival."
                )
                return {
                    "topic": "overview",
                    "text": f"{body}\n\n{self.numbers_text(lang)}\n\n{L('disclaimer')}",
                    "spoken": body,
                    "topics": topics,
                    "numbers": self.NUMBERS,
                }
        for key, topic in self.TOPICS.items():
            keywords = topic["keywords"]
            score = sum(1 for kw in keywords if normalize_arabic(kw.lower()) in needle)
            if score:
                title = topic["title_ar"] if lang == "ar" else topic["title_en"]
                steps = topic["steps_ar"] if lang == "ar" else topic["steps_en"]
                lines = "\n".join(f"{index + 1}. {step}" for index, step in enumerate(steps))
                text = f"{title}\n\n{lines}\n\n{self.numbers_text(lang)}\n\n{L('disclaimer')}"
                return {
                    "topic": key,
                    "text": text,
                    "spoken": f"{title}. " + " ".join(steps[:3]),
                    "numbers": self.NUMBERS,
                }
        if lang == "ar":
            body = "لم أجد موضوعاً مطابقاً. المواضيع المتاحة: " + "، ".join(
                t["title_ar"] for t in self.TOPICS.values()
            )
        else:
            body = "No matching topic. Available: " + ", ".join(
                t["title_en"] for t in self.TOPICS.values()
            )
        return {"topic": "", "text": f"{body}\n\n{self.numbers_text(lang)}", "spoken": body}


# --------------------------------------------------------------------------
# Diagnostics and controlled recovery
# --------------------------------------------------------------------------
class DiagnosticsEngine:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def run_all(self) -> List[Dict[str, Any]]:
        ctx = self.ctx
        checks: List[Dict[str, Any]] = []

        def add(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})

        ok, detail = ctx.store.integrity()
        add("SQLite integrity", "ok" if ok else "fail", detail)
        counts = ctx.store.query_one(
            "SELECT (SELECT COUNT(*) FROM messages) AS m, (SELECT COUNT(*) FROM memories) AS mem, "
            "(SELECT COUNT(*) FROM chunks) AS c"
        ) or {}
        add("Local data", "ok", f"{counts.get('m', 0)} messages · {counts.get('mem', 0)} memories · {counts.get('c', 0)} chunks")

        storage = ctx.bridge.storage_info()
        free = int(storage.get("free") or 0)
        add(
            "Storage",
            "ok" if free > 500 * 1024 * 1024 else ("warn" if free > 100 * 1024 * 1024 else "fail"),
            f"{human_bytes(free)} free",
        )

        models = ctx.models.scan(force=True)
        if not models:
            add("Local model", "warn", "No GGUF model found (heuristic fallback active)")
        else:
            selected = ctx.models.selected()
            verified = ctx.models.verify(selected.path) if selected else {"ok": False, "error": "none selected"}
            add(
                "Local model",
                "ok" if verified.get("ok") else "warn",
                f"{selected.name if selected else '?'} · {verified.get('message', verified.get('error', ''))}",
            )
        add("Inference engine", "ok" if ctx.llm.engine_name.startswith("llama") else "warn", ctx.llm.status)

        voice_models = discover_vosk_models()
        add(
            "Voice models",
            "ok" if voice_models else "warn",
            ("Vosk: " + ", ".join(sorted(voice_models))) if voice_models else "No Vosk model directories",
        )
        add("Speech output", "ok" if ctx.tts.available else "warn", ctx.tts.status)
        add("Encryption", "ok" if ctx.store.crypto.available else "warn", ctx.store.crypto.status)

        skills = ctx.skills.list_skills() if ctx.skills else []
        unhealthy = [s for s in skills if s.get("health") not in ("healthy", "disabled")]
        add(
            "Skill registry",
            "ok" if skills and not unhealthy else ("warn" if skills else "fail"),
            f"{len(skills)} skills · {len(unhealthy)} unhealthy",
        )

        deps = []
        for module in ("pypdf", "requests", "arabic_reshaper", "bidi", "plyer"):
            try:
                __import__(module)
                deps.append(f"{module}:ok")
            except ImportError:
                deps.append(f"{module}:missing")
        add("Python dependencies", "ok" if all(d.endswith(":ok") for d in deps) else "warn", ", ".join(deps))

        add("Android bridge", "ok" if ctx.bridge.available else "warn", ctx.bridge.reason)
        measure = ctx.resources.measure(force=True)
        add(
            "Memory pressure",
            "ok" if measure.get("ram_percent", 0) < 80 else "warn",
            f"{measure.get('ram_percent')}% used · {measure.get('thermal')} thermal",
        )
        network = measure.get("network", {})
        add(
            "Network",
            "ok" if network.get("online") else "warn",
            f"{'online' if network.get('online') else 'offline'} · mode {ctx.config.online_mode}",
        )
        try:
            error_size = ERROR_LOG_PATH.stat().st_size if ERROR_LOG_PATH.exists() else 0
            add("Error log", "ok" if error_size < 1_000_000 else "warn", f"{human_bytes(error_size)}")
        except OSError:
            add("Error log", "warn", "unreadable")
        return checks

    @staticmethod
    def report_text(checks: Sequence[Dict[str, Any]]) -> str:
        symbols = {"ok": "✓", "warn": "!", "fail": "✗"}
        lines = [f"{symbols.get(c['status'], '?')} {c['name']}: {c['detail']}" for c in checks]
        return "Diagnostics:\n" + "\n".join(lines)

    def export_report(self) -> Dict[str, Any]:
        checks = self.run_all()
        body = "\n".join(
            f"- [{c['status'].upper()}] {c['name']}: {c['detail']}" for c in checks
        )
        path = EXPORT_DIR / f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        content = (
            f"# Boshra diagnostics\n\n_{now_iso()} · v{APP_VERSION} · {self.ctx.config.profile}_\n\n"
            f"{body}\n"
        )
        if not atomic_write_text(path, content, 0o644):
            return {"ok": False, "error": "Cannot write report"}
        return {"ok": True, "path": str(path), "checks": checks}

    def recover(self, action: str) -> Dict[str, Any]:
        ctx = self.ctx
        if action == "vacuum":
            ok = ctx.store.vacuum()
            return {"ok": ok, "message": "Database vacuumed" if ok else "Vacuum failed"}
        if action == "reindex":
            count = 0
            for row in ctx.store.query("SELECT id, text, enc FROM chunks"):
                text = ctx.store.unseal(row["text"], row["enc"])
                ctx.store.execute(
                    "UPDATE chunks SET vec=? WHERE id=?", (vec_to_blob(embed_text(text)), row["id"])
                )
                count += 1
            for row in ctx.store.query("SELECT id, content, enc FROM memories"):
                text = ctx.store.unseal(row["content"], row["enc"])
                ctx.store.execute(
                    "INSERT INTO memory_vectors(memory_id, vec) VALUES(?,?) "
                    "ON CONFLICT(memory_id) DO UPDATE SET vec=excluded.vec",
                    (row["id"], vec_to_blob(embed_text(text))),
                )
            ctx.store.audit("diagnostics", "reindex", str(count), "low")
            return {"ok": True, "message": f"Rebuilt {count} chunk vectors"}
        if action == "disable-unhealthy" and ctx.skills:
            disabled = ctx.skills.disable_unhealthy()
            return {"ok": True, "message": f"Disabled {disabled} unhealthy skills"}
        if action == "clear-cache":
            removed = 0
            for path in (PRIVATE_DIR / "cache").glob("*"):
                try:
                    if path.is_file():
                        path.unlink()
                        removed += 1
                    elif path.is_dir():
                        shutil.rmtree(path)
                        removed += 1
                except OSError:
                    continue
            return {"ok": True, "message": f"Cleared {removed} cache entries (user data untouched)"}
        if action == "encrypt":
            changed = ctx.store.migrate_encryption()
            return {"ok": True, "message": f"Encrypted {changed} existing rows"}
        return {"ok": False, "message": f"Unknown recovery action: {action}"}


# --------------------------------------------------------------------------
# Unified semantic recall (chat, memory, notes, tasks, events, vault)
# --------------------------------------------------------------------------
RECALL_KINDS = {
    "chat": ("Chat", "محادثة"),
    "memory": ("Memory", "ذاكرة"),
    "note": ("Note", "ملاحظة"),
    "task": ("Task", "مهمة"),
    "event": ("Event", "موعد"),
    "commitment": ("Follow-up", "متابعة"),
    "vault": ("Vault", "خزنة"),
}

# Bridging common Arabic query words to the English terms users store data in.
# Keys are normalized the same way tokenize() normalizes queries (ة→ه, ى→ي).
RECALL_SYNONYMS = {
    normalize_arabic(key): value for key, value in {
        "مهمة": "task", "مهام": "task tasks", "ملاحظة": "note", "ملاحظات": "note notes",
        "خطة": "plan", "خطط": "plan plans", "محادثة": "chat", "محادثات": "chat chats",
        "ذاكرة": "memory", "موعد": "event", "مواعيد": "event events",
        "متابعة": "followup follow up commitment", "خزنة": "vault",
        "ملف": "file", "ملفات": "file files", "مراجعة": "review",
        "موجز": "brief digest", "تقرير": "report", "مشروع": "project",
        "اجتماع": "meeting", "تذكير": "reminder", "شراء": "buy purchase",
    }.items()
}


class RecallIndex:
    """One embedding index over everything the user has stored locally."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store

    def reindex(self, message_limit: int = 600) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        items: List[Tuple[str, str, str, str, int, bytes, str]] = []
        stamp = now_iso()

        def push(kind: str, ref: str, title: str, body: str) -> None:
            body = (body or "").strip()
            if not body:
                return
            sealed, enc = self.store.seal(body[:2400])
            items.append((kind, ref, (title or "")[:160], sealed, enc, vec_to_blob(embed_text(body)), stamp))
            counts[kind] = counts.get(kind, 0) + 1

        rows = self.store.query(
            "SELECT m.id, m.role, m.content, m.enc, c.title FROM messages m "
            "JOIN conversations c ON c.id=m.convo_id ORDER BY m.id DESC LIMIT ?",
            (message_limit,),
        )
        for row in rows:
            body = self.store.unseal(row["content"], row["enc"])
            push("chat", f"m{row['id']}", f"{row['title'] or 'Conversation'} · {row['role']}", body)

        for row in self.store.query("SELECT id, category, content, enc FROM memories ORDER BY id DESC LIMIT 600"):
            push("memory", f"m{row['id']}", f"Memory · {row['category']}",
                 self.store.unseal(row["content"], row["enc"]))
        for row in self.store.query("SELECT id, title, body, enc FROM notes ORDER BY id DESC LIMIT 400"):
            push("note", f"n{row['id']}", row["title"] or "Note", self.store.unseal(row["body"], row["enc"]))
        for row in self.store.query("SELECT id, title, done, due FROM tasks ORDER BY id DESC LIMIT 400"):
            push("task", f"t{row['id']}", "Task · " + ("done" if row["done"] else "open"),
                 f"{row['title']} {row['due']}")
        for row in self.store.query("SELECT id, title, start, notes FROM events ORDER BY id DESC LIMIT 400"):
            push("event", f"e{row['id']}", "Event · " + str(row["start"])[:10],
                 f"{row['title']} {row['start']} {row['notes']}")
        for row in self.store.query("SELECT id, text FROM commitments WHERE status='open' ORDER BY id DESC LIMIT 200"):
            push("commitment", f"c{row['id']}", "Follow-up", row["text"])
        for row in self.store.query(
            "SELECT ch.id, ch.page, ch.text, ch.enc, d.title FROM chunks ch "
            "JOIN documents d ON d.id=ch.doc_id ORDER BY ch.id DESC LIMIT 800"
        ):
            page = f" · p.{row['page']}" if row["page"] else ""
            push("vault", f"v{row['id']}", f"Vault · {row['title']}{page}",
                 self.store.unseal(row["text"], row["enc"]))

        with self.store.lock, self.store.conn:
            self.store.conn.execute("DELETE FROM recall_items")
            if items:
                self.store.conn.executemany(
                    "INSERT INTO recall_items(kind, ref, title, body, enc, vec, updated) "
                    "VALUES(?,?,?,?,?,?,?)",
                    items,
                )
            self.store.conn.execute(
                "INSERT INTO app_state(key, value) VALUES('recall:last_index', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (stamp,),
            )
        self.store.audit("recall", "reindex", json.dumps(counts), "low")
        return counts

    def reindex_if_stale(self, max_age_seconds: int = 180) -> bool:
        last = parse_iso(self.store.state_get("recall:last_index", ""))
        if last and (datetime.now() - last).total_seconds() < max_age_seconds:
            return False
        self.reindex()
        return True

    def search(self, query: str, k: int = 10, kinds: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        tokens = list(tokenize(query))
        for token in list(tokens):
            extra = RECALL_SYNONYMS.get(token)
            if extra:
                tokens.extend(extra.split())
        expanded = " ".join(tokens)
        qvec = embed_text(expanded)
        q_tokens = set(tokens)
        hits: List[Dict[str, Any]] = []
        for row in self.store.query("SELECT * FROM recall_items"):
            if kinds and row["kind"] not in kinds:
                continue
            body = self.store.unseal(row["body"], row["enc"])
            vector = blob_to_vec(row["vec"])
            similarity = cosine(qvec, vector)
            haystack = f"{row['title'] or ''} {body}"
            overlap = len(q_tokens & set(tokenize(haystack))) / max(1, len(q_tokens))
            score = 0.45 * similarity + 0.55 * overlap
            if score < 0.08:
                continue
            snippet = self._snippet(body, q_tokens)
            hits.append({
                "kind": row["kind"], "ref": row["ref"], "title": row["title"],
                "body": body, "snippet": snippet, "score": round(float(score), 3),
            })
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[:max(1, k)]

    @staticmethod
    def _snippet(body: str, tokens: set) -> str:
        flat = " ".join(body.split())
        lowered = normalize_arabic(flat.lower())
        index = -1
        for token in tokens:
            index = lowered.find(token)
            if index >= 0:
                break
        if index < 0:
            return flat[:240] + ("…" if len(flat) > 240 else "")
        start = max(0, index - 90)
        end = min(len(flat), index + 150)
        return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")

    def render(self, query: str, k: int = 10) -> Tuple[str, List[Dict[str, Any]]]:
        hits = self.search(query, k=k)
        if not hits:
            return (
                T(f"No local matches for “{query}”. Import documents or save a few notes first.",
                  f"لا توجد نتائج محلية عن «{query}». استورد مستندات أو احفظ ملاحظات أولاً."),
                [],
            )
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for hit in hits:
            grouped.setdefault(hit["kind"], []).append(hit)
        lines = [T(f"Recall for “{query}”", f"استرجاع عن «{query}»")]
        for kind in ("chat", "memory", "note", "task", "event", "commitment", "vault"):
            if kind not in grouped:
                continue
            english, arabic = RECALL_KINDS.get(kind, (kind, kind))
            lines.append("\n" + T(english, arabic) + f" · {len(grouped[kind])}")
            for hit in grouped[kind][:4]:
                lines.append(f"• [{hit['title']}] {hit['snippet']}")
        sources = [{"title": hit["title"], "kind": hit["kind"], "score": hit["score"]} for hit in hits]
        return "\n".join(lines), sources


# --------------------------------------------------------------------------
# Encrypted backup / restore (portable PBKDF2 + HMAC-SHA256 stream)
# --------------------------------------------------------------------------
BACKUP_MAGIC = b"BOSHRA-BK1\n"
BACKUP_ITERATIONS = 120_000


class BackupManager:
    """Encrypted local snapshots that survive app-data loss and device moves."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store
        self.config = ctx.config
        self.backup_dir = PRIVATE_DIR / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.public_dir = PUBLIC_DIR / "backups"
        with contextlib.suppress(OSError):
            self.public_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # -- crypto ----------------------------------------------------------
    @staticmethod
    def _keys(password: str, salt: bytes) -> Tuple[bytes, bytes]:
        material = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                                       BACKUP_ITERATIONS, dklen=64)
        return material[:32], material[32:]

    @staticmethod
    def _xor(data: bytes, key: bytes, nonce: bytes) -> bytes:
        out = bytearray(len(data))
        offset = 0
        counter = 0
        while offset < len(data):
            block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
            for i, byte in enumerate(block):
                if offset + i >= len(data):
                    break
                out[offset + i] = data[offset + i] ^ byte
            offset += len(block)
            counter += 1
        return bytes(out)

    def encrypt_bytes(self, data: bytes, password: str) -> bytes:
        salt, nonce = os.urandom(16), os.urandom(16)
        enc_key, mac_key = self._keys(password, salt)
        header = BACKUP_MAGIC + salt + nonce + struct.pack("<Q", len(data))
        cipher = self._xor(data, enc_key, nonce)
        tag = hmac.new(mac_key, header + cipher, hashlib.sha256).digest()
        return header + cipher + tag

    def decrypt_bytes(self, blob: bytes, password: str) -> bytes:
        if not blob.startswith(BACKUP_MAGIC):
            raise ValueError("Not a Boshra backup")
        offset = len(BACKUP_MAGIC)
        salt = blob[offset:offset + 16]
        nonce = blob[offset + 16:offset + 32]
        size = struct.unpack("<Q", blob[offset + 32:offset + 40])[0]
        header = blob[:offset + 40]
        cipher = blob[offset + 40:offset + 40 + size]
        tag = blob[offset + 40 + size:offset + 40 + size + 32]
        if len(cipher) != size or len(tag) != 32:
            raise ValueError("Backup is truncated")
        enc_key, mac_key = self._keys(password, salt)
        expected = hmac.new(mac_key, header + cipher, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, tag):
            raise ValueError("Wrong password or corrupted backup")
        return self._xor(cipher, enc_key, nonce)

    def _device_password(self) -> str:
        key_path = PRIVATE_DIR / "backup.key"
        try:
            if key_path.exists():
                return key_path.read_text(encoding="utf-8").strip()
            key = os.urandom(24).hex()
            key_path.write_text(key, encoding="utf-8")
            with contextlib.suppress(OSError):
                os.chmod(key_path, 0o600)
            return key
        except OSError as exc:
            LOG.warning(f"backup key unavailable: {exc}")
            return f"boshra-{uuid.getnode()}-{DB_PATH.stat().st_size if DB_PATH.exists() else 0}"

    # -- index -----------------------------------------------------------
    def _index(self) -> List[Dict[str, Any]]:
        try:
            data = json.loads(self.store.state_get("backup:index", "[]"))
            return data if isinstance(data, list) else []
        except (ValueError, TypeError):
            return []

    def _save_index(self, entries: List[Dict[str, Any]]) -> None:
        self.store.state_set("backup:index", json.dumps(entries[-30:], ensure_ascii=False))

    def list_backups(self) -> List[Dict[str, Any]]:
        entries = []
        for item in self._index():
            path = Path(str(item.get("path", "")))
            if path.exists():
                entries.append({**item, "size": path.stat().st_size})
        known = {str(Path(str(item.get("path", ""))).resolve()) for item in entries}
        for path in sorted(self.backup_dir.glob("*.boshra")):
            if str(path.resolve()) in known:
                continue
            entries.append({
                "path": str(path), "created": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "label": "backup", "mode": "unknown", "size": path.stat().st_size,
            })
        entries.sort(key=lambda item: str(item.get("created", "")), reverse=True)
        return entries

    # -- create / restore ------------------------------------------------
    def create(self, password: str = "", label: str = "manual") -> Dict[str, Any]:
        with self._lock:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            portable = bool(password)
            password = password or self._device_password()
            with tempfile.TemporaryDirectory(prefix="boshra_backup_") as temp:
                work = Path(temp)
                db_copy = work / "boshra.db"
                with self.store.lock:
                    destination = sqlite3.connect(str(db_copy))
                    self.store.conn.backup(destination)
                    destination.close()
                if CONFIG_PATH.exists():
                    shutil.copy2(CONFIG_PATH, work / "config.json")
                if SKILLS_DIR.exists():
                    shutil.copytree(SKILLS_DIR, work / "skills", dirs_exist_ok=True)
                vault_dir = PRIVATE_DIR / "vault"
                if vault_dir.exists():
                    shutil.copytree(vault_dir, work / "vault", dirs_exist_ok=True)
                manifest = {
                    "app": APP_NAME, "version": APP_VERSION, "build": BUILD_STAMP,
                    "created": now_iso(), "portable": portable,
                    "db_sha256": hashlib.sha256(db_copy.read_bytes()).hexdigest(),
                    "counts": {
                        "messages": int((self.store.query_one("SELECT COUNT(*) AS c FROM messages") or {}).get("c") or 0),
                        "memories": int((self.store.query_one("SELECT COUNT(*) AS c FROM memories") or {}).get("c") or 0),
                        "documents": int((self.store.query_one("SELECT COUNT(*) AS c FROM documents") or {}).get("c") or 0),
                        "plans": int((self.store.query_one("SELECT COUNT(*) AS c FROM plans") or {}).get("c") or 0),
                    },
                }
                (work / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
                archive = work / "snapshot.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    for item in work.iterdir():
                        if item.name != "snapshot.tar.gz":
                            tar.add(item, arcname=item.name)
                payload = self.encrypt_bytes(archive.read_bytes(), password)
            target = self.backup_dir / f"boshra-backup-{stamp}.boshra"
            if not atomic_write_bytes(target, payload):
                return {"ok": False, "error": "Could not write the backup file"}
            with contextlib.suppress(OSError, shutil.SameFileError):
                shutil.copy2(target, self.public_dir / target.name)
            entry = {
                "path": str(target), "created": now_iso(), "label": label[:60],
                "mode": "portable" if portable else "device", "size": target.stat().st_size,
            }
            entries = [item for item in self._index() if str(item.get("path")) != str(target)]
            entries.append(entry)
            self._save_index(entries)
            self.store.audit("backup", "create", f"{label} · {entry['mode']} · {entry['size']}", "medium")
            message = T(
                f"Encrypted backup created ({entry['mode']}): {target.name}",
                f"أُنشئت نسخة احتياطية مشفرة ({entry['mode']}): {target.name}",
            )
            return {"ok": True, "path": str(target), "entry": entry, "message": message}

    def restore(self, path: str, password: str = "") -> Dict[str, Any]:
        source = Path(path)
        if not source.exists():
            return {"ok": False, "error": "Backup file not found"}
        try:
            blob = source.read_bytes()
            plain = self.decrypt_bytes(blob, password or self._device_password())
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except OSError as exc:
            return {"ok": False, "error": f"Could not read the backup: {exc}"}
        with tempfile.TemporaryDirectory(prefix="boshra_restore_") as temp:
            work = Path(temp)
            archive = work / "snapshot.tar.gz"
            archive.write_bytes(plain)
            extracted = work / "files"
            extracted.mkdir()
            _safe_extract_tar(archive, extracted)
            db_file = extracted / "boshra.db"
            if not db_file.exists():
                return {"ok": False, "error": "Backup does not contain a database"}
            try:
                check = sqlite3.connect(str(db_file))
                result = check.execute("PRAGMA integrity_check").fetchone()
                check.close()
            except sqlite3.Error as exc:
                return {"ok": False, "error": f"Database check failed: {exc}"}
            if not result or str(result[0]).lower() != "ok":
                return {"ok": False, "error": "Backup database failed its integrity check"}
            staging = PRIVATE_DIR / "restore_staging"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            for name in ("boshra.db", "config.json", "manifest.json"):
                if (extracted / name).exists():
                    shutil.copy2(extracted / name, staging / name)
            for name in ("skills", "vault"):
                if (extracted / name).exists():
                    shutil.copytree(extracted / name, staging / name, dirs_exist_ok=True)
            marker = PRIVATE_DIR / "restore_pending.json"
            marker.write_text(json.dumps({
                "staged": str(staging), "source": str(source), "created": now_iso(),
            }, ensure_ascii=False), encoding="utf-8")
            self.store.audit("backup", "stage-restore", source.name, "high")
            return {
                "ok": True,
                "message": T(
                    "Backup verified and staged. Restart Boshra to apply it.",
                    "تم التحقق من النسخة وتجهيزها. أعد تشغيل بشرى لتطبيقها.",
                ),
                "requires_restart": True,
            }

    def maybe_auto_backup(self) -> Optional[Dict[str, Any]]:
        days = int(self.config.auto_backup_days or 0)
        if days <= 0:
            return None
        entries = [item for item in self._index() if item.get("mode") == "device"]
        if entries:
            last = parse_iso(str(entries[-1].get("created", "")))
            if last and (datetime.now() - last).total_seconds() < days * 86400:
                return None
        return self.create(label="automatic")


def atomic_write_bytes(path: Path, payload: bytes) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            with contextlib.suppress(OSError):
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    except OSError as exc:
        LOG.error(f"atomic binary write failed: {path}", exc)
        return False


def apply_pending_restore() -> bool:
    """Apply a staged restore before the database is opened."""
    marker = PRIVATE_DIR / "restore_pending.json"
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        staged = Path(str(info.get("staged", "")))
        if not staged.exists() or not (staged / "boshra.db").exists():
            raise ValueError("staged restore is incomplete")
        check = sqlite3.connect(str(staged / "boshra.db"))
        result = check.execute("PRAGMA integrity_check").fetchone()
        check.close()
        if not result or str(result[0]).lower() != "ok":
            raise ValueError("staged database failed integrity check")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        previous = PRIVATE_DIR / f"pre_restore_{stamp}"
        previous.mkdir(parents=True, exist_ok=True)
        for name in ("boshra.db", "boshra.db-wal", "boshra.db-shm", "config.json"):
            source = PRIVATE_DIR / name
            if source.exists():
                shutil.move(str(source), str(previous / name))
        skills = PRIVATE_DIR / "skills"
        vault = PRIVATE_DIR / "vault"
        if skills.exists():
            shutil.move(str(skills), str(previous / "skills"))
        if vault.exists():
            shutil.move(str(vault), str(previous / "vault"))
        for name in ("boshra.db", "config.json"):
            source = staged / name
            if source.exists():
                shutil.copy2(source, PRIVATE_DIR / name)
        for name in ("skills", "vault"):
            source = staged / name
            if source.exists():
                shutil.copytree(source, PRIVATE_DIR / name, dirs_exist_ok=True)
        shutil.rmtree(staged, ignore_errors=True)
        marker.unlink(missing_ok=True)
        LOG.info(f"restored backup staged at {stamp}; previous data in {previous}")
        return True
    except Exception as exc:  # noqa: BLE001
        LOG.error("pending restore failed", exc)
        _append_error_dump("apply-restore", traceback.format_exc())
        return False


# --------------------------------------------------------------------------
# Multi-step plan engine
# --------------------------------------------------------------------------
class PlanEngine:
    MAX_STEPS = 5

    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store

    def _clauses(self, goal: str) -> List[str]:
        cleaned = re.sub(r"^\s*(?:plan|خطط|خطة)\s*[:\-]?\s*", "", goal or "", flags=re.IGNORECASE)
        parts = re.split(
            r"\s*(?:;|\n|→|->|\bthen\b|\bafter that\b|\bبعدها\b|\bثم\b)\s*",
            cleaned, flags=re.IGNORECASE,
        )
        clauses = []
        for part in parts:
            part = re.sub(r"^\s*\d+[.)-]\s*", "", part).strip(" .")
            if len(part) >= 3:
                clauses.append(part[:180])
        return clauses[:self.MAX_STEPS]

    def propose(self, goal: str) -> Dict[str, Any]:
        goal = (goal or "").strip()
        if not goal:
            return {"ok": False, "error": "Plan needs a goal"}
        steps: List[Dict[str, Any]] = []
        for clause in self._clauses(goal):
            route = None
            with contextlib.suppress(Exception):
                route = self.ctx.orchestrator._route(clause) if self.ctx.orchestrator else None
            if not route:
                continue
            kind, payload = route
            if kind == "tool":
                steps.append({
                    "title": clause, "kind": "tool", "tool": payload.get("tool", ""),
                    "args": dict(payload.get("args") or {}), "status": "pending", "detail": "",
                })
            elif kind == "routine":
                steps.append({
                    "title": clause, "kind": "routine", "routine": payload.get("name", ""),
                    "status": "pending", "detail": "",
                })
            elif kind == "remember":
                steps.append({
                    "title": clause, "kind": "tool", "tool": "memory_add",
                    "args": {"content": payload.get("content", clause)},
                    "status": "pending", "detail": "",
                })
        if not steps:
            return {
                "ok": False,
                "error": T(
                    "I could not turn that into tool steps. Try naming the actions, e.g. “plan: list tasks, then add a note”.",
                    "تعذر تحويل الطلب إلى خطوات أدوات. جرّب تسمية الإجراءات، مثل: «خطة: اعرض المهام ثم أضف ملاحظة».",
                ),
            }
        plan = self.create(goal, steps)
        return {"ok": True, "plan": plan}

    def create(self, goal: str, steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        conversation_id = int(self.ctx.config.last_conversation_id or 0)
        plan_id = self.store.execute(
            "INSERT INTO plans(convo_id, goal, status, steps, created, updated) VALUES(?,?,?,?,?,?)",
            (conversation_id, goal[:400], "proposed", json.dumps(list(steps), ensure_ascii=False),
             now_iso(), now_iso()),
        )
        row = self.get(plan_id)
        self.store.audit("plan", "propose", goal[:220], "low")
        return row or {}

    def get(self, plan_id: int) -> Optional[Dict[str, Any]]:
        row = self.store.query_one("SELECT * FROM plans WHERE id=?", (plan_id,))
        if not row:
            return None
        with contextlib.suppress(ValueError, TypeError):
            row["steps"] = json.loads(row["steps"])
        return row

    def recent(self, limit: int = 8) -> List[Dict[str, Any]]:
        rows = self.store.query("SELECT * FROM plans ORDER BY id DESC LIMIT ?", (limit,))
        for row in rows:
            with contextlib.suppress(ValueError, TypeError):
                row["steps"] = json.loads(row["steps"])
        return rows

    def _save(self, plan_id: int, status: str, steps: Sequence[Dict[str, Any]], result: str = "") -> None:
        self.store.execute(
            "UPDATE plans SET status=?, steps=?, result=?, updated=? WHERE id=?",
            (status, json.dumps(list(steps), ensure_ascii=False), result[:4000], now_iso(), plan_id),
        )

    def run(self, plan_id: int,
            on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        plan = self.get(plan_id)
        if not plan:
            return {"ok": False, "error": "Plan not found"}
        steps = list(plan.get("steps") or [])
        if not steps:
            return {"ok": False, "error": "Plan has no steps"}
        self._save(plan_id, "running", steps)
        stop_event = getattr(self.ctx.orchestrator, "stop_event", None)
        cancelled = False
        for index, step in enumerate(steps):
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break
            step["status"] = "running"
            self._save(plan_id, "running", steps)
            if on_step:
                on_step(index, step)
            detail = ""
            ok = True
            try:
                if step.get("kind") == "routine":
                    outcome = self.ctx.automation.run_by_name(str(step.get("routine", ""))) if self.ctx.automation else {"ok": False}
                    ok = bool(outcome.get("ok"))
                    detail = str(outcome.get("message") or outcome.get("error") or "")[:500]
                elif step.get("kind") == "tool":
                    outcome = self.ctx.tools.execute(str(step.get("tool", "")), dict(step.get("args") or {}))
                    ok = bool(outcome.ok)
                    detail = (outcome.render() if outcome.ok else str(outcome.error or "failed"))[:500]
                else:
                    detail = "noted"
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = f"{type(exc).__name__}: {exc}"[:500]
            step["status"] = "done" if ok else "failed"
            step["detail"] = detail
            self._save(plan_id, "running", steps)
            if on_step:
                on_step(index, step)
            if not ok:
                break
        status = "cancelled" if cancelled else ("done" if all(s.get("status") == "done" for s in steps) else "partial")
        lines = [T(f"Plan: {plan['goal']}", f"الخطة: {plan['goal']}")]
        for step in steps:
            mark = {"done": "✓", "failed": "✕", "running": "…", "pending": "○"}.get(step.get("status", "pending"), "•")
            lines.append(f"{mark} {step.get('title', '')}" + (f"\n   {step['detail'][:240]}" if step.get("detail") else ""))
        summary = "\n".join(lines)
        self._save(plan_id, status, steps, summary)
        with contextlib.suppress(Exception):
            lang = self.ctx.orchestrator._lang_for(plan["goal"])
            self.ctx.orchestrator._save_message("assistant", summary, lang)
        self.store.audit("plan", "run", f"#{plan_id} {status}", "medium")
        return {"ok": status == "done", "status": status, "summary": summary, "steps": steps, "plan_id": plan_id}


# --------------------------------------------------------------------------
# Proactive digests, quiet hours and commitment follow-ups
# --------------------------------------------------------------------------
class ProactiveEngine:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store
        self.config = ctx.config

    @staticmethod
    def detect_commitment(text: str) -> str:
        match = re.search(
            r"(?<!\w)(?:i will|i'll|i am going to|i'm going to|remind me to)\s+(.{3,140})"
            r"|(?<!\w)(?:سأ|سوف|سأقوم|ساقوم|لازم|عليّ أن|علي ان)\s*(.{3,140})",
            text or "", re.IGNORECASE,
        )
        return match.group(0).strip(" .")[:180] if match else ""

    def note_commitment(self, text: str) -> Optional[str]:
        if not self.config.commitment_detection or looks_sensitive(text):
            return None
        phrase = self.detect_commitment(text)
        if not phrase:
            return None
        existing = self.store.query_one(
            "SELECT id FROM commitments WHERE status='open' AND text=? COLLATE NOCASE", (phrase,)
        )
        if existing:
            return None
        self.store.execute(
            "INSERT INTO commitments(text, source, status, created) VALUES(?,?,?,?)",
            (phrase, "chat", "open", now_iso()),
        )
        return phrase

    def open_commitments(self, limit: int = 8) -> List[Dict[str, Any]]:
        return self.store.query(
            "SELECT * FROM commitments WHERE status='open' ORDER BY id DESC LIMIT ?", (limit,)
        )

    def resolve_commitment(self, commitment_id: int, status: str = "done") -> bool:
        status = status if status in ("done", "dismissed") else "done"
        self.store.execute("UPDATE commitments SET status=? WHERE id=?", (status, int(commitment_id)))
        return True

    def _lines(self, kind: str) -> List[str]:
        now = datetime.now()
        lines: List[str] = []
        tasks = self.store.query(
            "SELECT title, due FROM tasks WHERE done=0 ORDER BY CASE WHEN due='' THEN 1 ELSE 0 END, due LIMIT 8"
        )
        if tasks:
            lines.append(T("Open tasks", "مهام مفتوحة"))
            for task in tasks:
                due = f" · {task['due'][:16]}" if task["due"] else ""
                lines.append(f"• {task['title']}{due}")
        soon = (now + timedelta(hours=24)).isoformat(timespec="seconds")
        reminders = self.store.query(
            "SELECT title, at FROM reminders WHERE fired=0 AND at<=? ORDER BY at LIMIT 8", (soon,)
        )
        if reminders:
            lines.append("\n" + T("Next 24 hours", "الـ24 ساعة القادمة"))
            for reminder in reminders:
                lines.append(f"• {reminder['title']} · {reminder['at'][:16]}")
        events = self.store.query(
            "SELECT title, start FROM events WHERE substr(start,1,10)=? ORDER BY start LIMIT 6",
            (now.date().isoformat(),),
        )
        if events:
            lines.append("\n" + T("Today's calendar", "مواعيد اليوم"))
            for event in events:
                lines.append(f"• {event['title']} · {event['start'][11:16]}")
        commitments = self.open_commitments(6)
        if commitments:
            lines.append("\n" + T("Follow-ups", "متابعات"))
            for item in commitments:
                lines.append(f"• {item['text']}")
        plans = self.store.query("SELECT id, goal, status FROM plans ORDER BY id DESC LIMIT 3")
        if plans:
            lines.append("\n" + T("Recent plans", "خطط حديثة"))
            for plan in plans:
                lines.append(f"• #{plan['id']} {plan['goal'][:80]} · {plan['status']}")
        if not lines:
            lines.append(T("Nothing urgent is waiting. A good moment for deep work.",
                           "لا شيء عاجل ينتظر. وقت مناسب للعمل العميق."))
        header = T("Morning brief", "موجز الصباح") if kind == "morning" else T("Evening review", "مراجعة المساء")
        return [header, f"{now.strftime('%Y-%m-%d %H:%M')}", ""] + lines

    def build(self, kind: str = "morning") -> Dict[str, Any]:
        body = "\n".join(self._lines(kind))
        digest_id = self.store.execute(
            "INSERT INTO digests(kind, day, body, created) VALUES(?,?,?,?)",
            (kind, datetime.now().date().isoformat(), body, now_iso()),
        )
        return {"id": digest_id, "kind": kind, "body": body, "created": now_iso()}

    def latest(self) -> Optional[Dict[str, Any]]:
        return self.store.query_one("SELECT * FROM digests ORDER BY id DESC LIMIT 1")

    def check(self, now: Optional[datetime] = None, force: bool = False) -> Optional[Dict[str, Any]]:
        if not force and not self.config.proactive_enabled:
            return None
        now = now or datetime.now()
        kind = ""
        if force:
            kind = "manual"
        elif now.hour >= int(self.config.evening_hour):
            kind = "evening"
        elif now.hour >= int(self.config.morning_hour):
            kind = "morning"
        if not kind:
            return None
        key = f"proactive:{kind}:{now.date().isoformat()}"
        if not force and self.store.state_get(key, ""):
            return None
        digest = self.build(kind)
        self.store.state_set(key, "1")
        quiet = in_quiet_hours(now, self.config.quiet_start, self.config.quiet_end)
        digest["quiet"] = quiet
        if not quiet:
            with contextlib.suppress(Exception):
                self.ctx.bridge.notify(T("Boshra brief", "موجز بشرى"), digest["body"][:220])
            self.ctx.emit("proactive", digest=digest)
        self.store.audit("proactive", "digest", kind, "low")
        return digest


# --------------------------------------------------------------------------
# Conversation compaction and context budgeting
# --------------------------------------------------------------------------
class ConversationCompactor:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store

    def _rows(self, conversation_id: int) -> List[Dict[str, Any]]:
        rows = self.store.query(
            "SELECT id, role, content, enc, ts FROM messages WHERE convo_id=? ORDER BY id",
            (conversation_id,),
        )
        for row in rows:
            row["content"] = self.store.unseal(row["content"], row["enc"])
        return rows

    def compact(self, conversation_id: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
        conversation_id = int(conversation_id or self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return {"ok": False, "error": "No conversation"}
        rows = self._rows(conversation_id)
        keep = 8
        older = rows[:-keep] if len(rows) > keep else []
        if len(older) < 4 and not force:
            return {"ok": False, "error": "Not enough history to compact"}
        if not older:
            return {"ok": False, "error": "Nothing older to summarize"}
        lines: List[str] = []
        for row in older:
            gist = split_sentences(row["content"], 1)
            text = (gist[0] if gist else row["content"]).strip().replace("\n", " ")[:220]
            if text:
                lines.append(f"{row['role']}: {text}")
        summary = "\n".join(lines)
        if len(summary) > 600:
            summary = summary[:340] + "\n…\n" + summary[-220:]
        sealed, enc = self.store.seal(summary)
        self.store.execute(
            "INSERT INTO conversation_summaries(convo_id, summary, enc, through_id, updated) "
            "VALUES(?,?,?,?,?) ON CONFLICT(convo_id) DO UPDATE SET "
            "summary=excluded.summary, enc=excluded.enc, through_id=excluded.through_id, updated=excluded.updated",
            (conversation_id, sealed, enc, int(older[-1]["id"]), now_iso()),
        )
        self.store.audit("chat", "compact", f"convo {conversation_id} · {len(older)} turns", "low")
        return {"ok": True, "summary": summary, "through_id": older[-1]["id"], "count": len(older)}

    def maybe_compact(self, conversation_id: Optional[int] = None) -> bool:
        if not self.ctx.config.compaction_enabled:
            return False
        conversation_id = int(conversation_id or self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return False
        summary = self.summary_row(conversation_id)
        through = int((summary or {}).get("through_id") or 0)
        count = self.store.query_one(
            "SELECT COUNT(*) AS c FROM messages WHERE convo_id=? AND id>?",
            (conversation_id, through),
        ) or {}
        if int(count.get("c") or 0) <= 16:
            return False
        return bool(self.compact(conversation_id).get("ok"))

    def summary_row(self, conversation_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        conversation_id = int(conversation_id or self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return None
        row = self.store.query_one(
            "SELECT * FROM conversation_summaries WHERE convo_id=?", (conversation_id,)
        )
        if row:
            row["summary"] = self.store.unseal(row["summary"], row["enc"])
        return row

    def history_block(self, recent: int = 8, exclude_last: bool = True) -> str:
        conversation_id = int(self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return ""
        summary = self.summary_row(conversation_id)
        window = recent if summary else max(recent, 16)
        rows = self.store.query(
            "SELECT id, role, content, enc FROM messages WHERE convo_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, window + (1 if exclude_last else 0)),
        )
        ordered = list(reversed(rows))
        if exclude_last and ordered:
            ordered = ordered[:-1]
        lines: List[str] = []
        through = int((summary or {}).get("through_id") or 0)
        if summary and summary.get("summary"):
            lines.append("[SUMMARY OF EARLIER TURNS]\n" + str(summary["summary"])[:600])
        for row in ordered:
            if int(row["id"]) <= through:
                continue
            content = self.store.unseal(row["content"], row["enc"]).replace("\n", " ")[:500]
            lines.append(f"{row['role'].upper()}: {content}")
        return "\n".join(lines)

    def usage(self) -> Dict[str, Any]:
        profile = self.ctx.resources.profile
        limit = int(profile.get("n_ctx") or 2048)
        used = approximate_tokens(self.history_block(exclude_last=False)) + 320
        return {"used": used, "limit": limit, "percent": min(100, round(used * 100 / max(1, limit)))}


# --------------------------------------------------------------------------
# Tool registry wiring
# --------------------------------------------------------------------------
def build_tool_registry(ctx: AppContext) -> ToolRegistry:
    registry = ToolRegistry(ctx)

    def add(
        name: str,
        description: str,
        handler: Callable[[AppContext, Dict[str, Any]], ToolResult],
        permissions: Tuple[str, ...] = (),
        risk: str = "low",
        destructive: bool = False,
        needs_network: bool = False,
        preview: Optional[Callable[[AppContext, Dict[str, Any]], ToolResult]] = None,
    ) -> None:
        registry.register(
            ToolSpec(name, description, handler, permissions, risk, destructive, needs_network, preview)
        )

    add("daily_briefing", "Local daily agenda: tasks, reminders and calendar", tool_daily_briefing)
    add("time_date", "Current local time and date", tool_time)
    add("device_info", "Device model, memory and CPU", tool_device_info, ("device.info",))
    add("battery", "Battery level and charging state", tool_battery, ("device.info",))
    add("storage", "Free and total storage", tool_storage, ("device.info",))
    add("open_app", "Launch an installed Android application", tool_open_app, ("android.app.open",), "medium")
    add("file_search", "Search approved Boshra folders", tool_file_search, ("files.search",))
    add("read_document", "Read a PDF/DOCX/text document", tool_read_document, ("documents.read",))
    add("vault_import", "Import and index a document", tool_vault_import, ("documents.read",))
    add("vault_list", "List indexed documents", tool_vault_list, ("documents.read",))
    add("vault_search", "Search indexed passages", tool_vault_search, ("documents.read",))
    add("vault_answer", "Answer from vault passages with sources", tool_vault_answer, ("documents.read",))
    add("vault_summarize", "Extractive summary of a document", tool_vault_summarize, ("documents.read",))
    add("vault_export", "Export a document summary", tool_vault_export, ("documents.write",), "medium")
    add("vault_delete", "Delete a document from the vault", tool_vault_delete, ("documents.delete",), "high")
    add("memory_add", "Remember a fact or preference", tool_memory_add, ("memory.write",))
    add("memory_search", "Search local memory", tool_memory_search, ("memory.read",))
    add("memory_delete", "Delete one memory", tool_memory_delete, ("memory.write",), "medium")
    add("task_add", "Add a task", tool_task_add)
    add("task_list", "List tasks", tool_task_list)
    add("task_done", "Complete a task", tool_task_done)
    add("note_add", "Save a note", tool_note_add)
    add("note_list", "List notes", tool_note_list)
    add("reminder_add", "Create a local reminder", tool_reminder_add)
    add("reminder_list", "List reminders", tool_reminder_list)
    add("event_add", "Add a local calendar entry", tool_event_add)
    add("event_list", "List local calendar entries", tool_event_list)
    add("focus_start", "Start a focus timer", tool_focus_timer)
    add("focus_stop", "Stop the focus timer", tool_focus_stop)
    add("emergency", "Offline emergency and survival guidance", tool_emergency, ())
    add("web_search", "Search the web (DuckDuckGo HTML)", tool_web_search, ("network.access",), "medium", needs_network=True)
    add("news", "Fetch RSS headlines", tool_news, ("network.access",), "medium", needs_network=True)
    add("weather", "Weather for a city", tool_weather, ("network.access",), "medium", needs_network=True)
    add("rss", "Fetch a user-supplied RSS feed", tool_rss, ("network.access",), "medium", needs_network=True)
    add("translate", "Translate AR/EN locally or with loaded model", tool_translate)
    add("code_explain", "Static explanation of Python code", tool_code_explain)
    add("secure_review", "Pattern-based secure coding review", tool_secure_review)
    add("refactor", "AST-based refactoring suggestions", tool_refactor)
    add("test_generate", "Generate pytest skeleton from AST", tool_test_generate)
    add("debug", "Analyse a Python traceback", tool_debug)
    add("report", "Write a markdown report", tool_report, ("documents.write",), "medium")
    add(
        "organize_files", "Organise files in BoshraAI workspace (preview first)",
        tool_organize_apply, ("files.move",), "high", destructive=True,
        preview=tool_organize_preview,
    )
    add("clipboard_read", "Read the clipboard", tool_clipboard_read, ("clipboard.read",), "high")
    add("clipboard_write", "Write text to the clipboard", tool_clipboard_write, ("clipboard.write",))
    add("notify", "Show a local notification", tool_notify, ("notifications.show",))
    add("theme_set", "Change the visual theme", tool_theme)
    add("profile_set", "Change LITE/BALANCED/PERFORMANCE profile", tool_profile)
    add("skill_list", "List installed skills", tool_skill_list)
    add("skill_execute", "Execute a registered skill", tool_skill_execute, ("skills.execute",), "medium")
    add("workflow_run", "Run a saved routine", tool_workflow_run, ("skills.execute",), "medium")
    add("math", "Evaluate a local arithmetic expression", tool_math)
    def run_diagnostics(c: AppContext, _args: Dict[str, Any]) -> ToolResult:
        checks = c.diagnostics.run_all() if c.diagnostics else []
        return ToolResult(True, DiagnosticsEngine.report_text(checks), data={"checks": checks})

    add("diagnostics", "Run local system diagnostics", run_diagnostics, ("device.info",))
    add("recall", "Semantic recall across chats, notes, tasks, memory and vault", tool_recall)
    add("recall_reindex", "Rebuild the local semantic recall index", tool_recall_reindex)
    add("plan_create", "Turn a goal into a multi-step tool plan", tool_plan_create)
    add("plan_run", "Execute the latest or a chosen plan step by step", tool_plan_run, ("skills.execute",), "medium")
    add("plan_list", "List recent plans and their status", tool_plan_list)
    add("backup_create", "Create an encrypted local backup", tool_backup_create, ("documents.write",), "medium")
    add("backup_list", "List encrypted backups", tool_backup_list)
    add(
        "backup_restore", "Restore an encrypted backup (applies on restart)",
        tool_backup_restore, ("files.move",), "high", destructive=True,
        preview=lambda c, a: ToolResult(
            True,
            T("Restore the selected backup? Current data is kept aside, and Boshra restarts.",
              "استعادة النسخة المحددة؟ ستُحفظ البيانات الحالية جانباً وستُعاد بشرى التشغيل."),
        ),
    )
    add("digest", "Build a morning, evening or manual chief-of-staff brief", tool_digest)
    add("commitments", "List or resolve detected follow-ups", tool_commitments)
    add("compact", "Summarize older turns to free context space", tool_compact)
    add("float", "Control the floating mini assistant", tool_float)
    return registry


# --------------------------------------------------------------------------
# Skill engine
# --------------------------------------------------------------------------
SKILL_CATEGORIES = ("productivity", "documents", "system", "intelligence", "development", "online", "custom")
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
ALLOWED_SKILL_DEPENDENCIES = {
    "pypdf", "requests", "arabic_reshaper", "bidi", "plyer", "zipfile", "sqlite3",
    "json", "re", "math", "datetime", "pathlib", "typing", "dataclasses", "csv", "xml",
}


@dataclass
class SkillResult:
    success: bool
    skill_id: str
    action: str = ""
    result: Any = None
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "skill_id": self.skill_id,
            "action": self.action,
            "result": self.result,
            "message": self.message,
            "error": self.error,
        }


def _version_tuple(version: str) -> Tuple[int, int, int]:
    match = SEMVER_RE.match(version or "")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else (0, 0, 0)


def validate_manifest(
    raw: Any, skill_dir: Optional[Path] = None, known_tools: Optional[Iterable[str]] = None, builtin: bool = False
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(raw, dict):
        return None, ["Manifest must be a JSON object"]
    data = dict(raw)
    for key in ("id", "name", "version", "category"):
        if not str(data.get(key, "")).strip():
            errors.append(f"Missing required field: {key}")
    skill_id = str(data.get("id", "")).strip()
    if skill_id and not SKILL_ID_RE.match(skill_id):
        errors.append("id must look like 'author.skill_name' (lowercase, dots/underscores)")
    if not SEMVER_RE.match(str(data.get("version", ""))):
        errors.append("version must be semantic (e.g. 1.0.0)")
    category = str(data.get("category", "")).strip()
    if category and category not in SKILL_CATEGORIES:
        errors.append(f"category must be one of {', '.join(SKILL_CATEGORIES)}")
    risk = str(data.get("risk_level", "low")).lower()
    if risk not in ("low", "medium", "high"):
        errors.append("risk_level must be low, medium or high")
        risk = "low"
    languages = data.get("languages") or ["en"]
    if not isinstance(languages, list) or any(lang not in ("ar", "en") for lang in languages):
        errors.append("languages must be a list containing 'ar' and/or 'en'")
        languages = ["en"]
    permissions = data.get("permissions") or []
    optional_permissions = data.get("optional_permissions") or []
    for permission in list(permissions) + list(optional_permissions):
        if permission not in PERMISSION_CATALOG:
            errors.append(f"Unknown permission: {permission}")
    # risk must not understate its permissions
    highest = max(
        ["low"] + [PERMISSION_CATALOG[p][0] for p in permissions if p in PERMISSION_CATALOG],
        key=lambda r: {"low": 0, "medium": 1, "high": 2}.get(r, 0),
    )
    if {"low": 0, "medium": 1, "high": 2}[highest] > {"low": 0, "medium": 1, "high": 2}[risk]:
        risk = highest
    internet_required = bool(data.get("internet_required", False))
    if internet_required and "network.access" not in permissions:
        errors.append("internet_required skills must declare the network.access permission")
    dependencies = data.get("dependencies") or []
    if not isinstance(dependencies, list):
        errors.append("dependencies must be a list")
        dependencies = []
    for dependency in dependencies:
        if str(dependency) not in ALLOWED_SKILL_DEPENDENCIES:
            errors.append(f"Dependency not supported by the Android build: {dependency}")
    entry_point = str(data.get("entry_point", "")).strip()
    if entry_point:
        if os.path.isabs(entry_point) or ".." in Path(entry_point).parts:
            errors.append("entry_point must be a relative path inside the skill folder")
        elif not entry_point.endswith(".py"):
            errors.append("entry_point must be a .py file")
        elif skill_dir is not None and not (skill_dir / entry_point).is_file():
            errors.append(f"entry_point not found: {entry_point}")
    steps = data.get("steps") or []
    if steps and not isinstance(steps, list):
        errors.append("steps must be a list")
        steps = []
    tool_names = set(known_tools or [])
    clean_steps: List[Dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not str(step.get("tool", "")).strip():
            errors.append(f"step {index + 1}: needs a 'tool' string")
            continue
        if tool_names and str(step["tool"]) not in tool_names:
            errors.append(f"step {index + 1}: unknown tool '{step['tool']}'")
        args = step.get("args", {})
        if not isinstance(args, dict):
            errors.append(f"step {index + 1}: args must be an object")
            args = {}
        clean_steps.append({"tool": str(step["tool"]), "args": args})
    if not entry_point and not clean_steps and not builtin:
        errors.append("Provide either an entry_point (skill.py) or declarative steps")
    commands_raw = data.get("commands") or []
    commands: List[Dict[str, Any]] = []
    if isinstance(commands_raw, dict):
        commands_raw = [
            {"name": name, "description": value if isinstance(value, str) else (value or {}).get("description", "")}
            for name, value in commands_raw.items()
        ]
    if commands_raw and not isinstance(commands_raw, list):
        errors.append("commands must be a list or object")
    for item in commands_raw if isinstance(commands_raw, list) else []:
        if isinstance(item, str):
            commands.append({"name": item, "description": ""})
        elif isinstance(item, dict) and str(item.get("name", "")).strip():
            command = {
                "name": str(item["name"]).strip(),
                "description": str(item.get("description", ""))[:300],
            }
            if isinstance(item.get("steps"), list):
                command_steps: List[Dict[str, Any]] = []
                for index, step in enumerate(item["steps"]):
                    if not isinstance(step, dict) or not str(step.get("tool", "")).strip():
                        errors.append(f"command {command['name']} step {index + 1} invalid")
                        continue
                    if tool_names and str(step["tool"]) not in tool_names:
                        errors.append(f"command {command['name']}: unknown tool '{step['tool']}'")
                    command_steps.append({"tool": str(step["tool"]), "args": step.get("args", {}) or {}})
                command["steps"] = command_steps
            commands.append(command)
        else:
            errors.append("Each command needs a name")
    if not commands:
        commands = [{"name": "run", "description": "Run the skill"}]
    min_version = str(data.get("min_app_version", "4.0.0"))
    if not SEMVER_RE.match(min_version):
        errors.append("min_app_version must be semantic")
    elif _version_tuple(min_version) > _version_tuple(APP_VERSION):
        errors.append(f"Requires app version {min_version}; this build is {APP_VERSION}")
    config_schema = data.get("config_schema") or {}
    if not isinstance(config_schema, dict):
        errors.append("config_schema must be an object")
    normalized = {
        "id": skill_id,
        "name": str(data.get("name", skill_id))[:80],
        "version": str(data.get("version", "0.0.0")),
        "description": str(data.get("description", ""))[:600],
        "author": str(data.get("author", "unknown"))[:80],
        "category": category or "custom",
        "entry_point": entry_point,
        "languages": languages,
        "permissions": [p for p in permissions if p in PERMISSION_CATALOG],
        "optional_permissions": [p for p in optional_permissions if p in PERMISSION_CATALOG],
        "dependencies": [str(d) for d in dependencies],
        "commands": commands,
        "steps": clean_steps,
        "min_app_version": min_version,
        "offline": bool(data.get("offline", not internet_required)),
        "internet_required": internet_required,
        "risk_level": risk,
        "config_schema": config_schema,
    }
    return (normalized if not errors else None), errors


def _skill_tool(ctx: AppContext, skill_id: str, action: str, tool: str, args: Dict[str, Any]) -> SkillResult:
    result = ctx.tools.execute(tool, args)
    return SkillResult(result.ok, skill_id, action, result.data, result.message, result.error)


def _skill_message(skill_id: str, action: str, message: str, data: Any = None) -> SkillResult:
    return SkillResult(True, skill_id, action, data or {}, message)


def _skill_error(skill_id: str, action: str, error: str) -> SkillResult:
    return SkillResult(False, skill_id, action, {}, error, error)


# -- built-in handlers -------------------------------------------------------
def skill_task_manager(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.task_manager"
    if command == "add":
        return _skill_tool(ctx, sid, command, "task_add", {"title": data.get("title", "")})
    if command == "done":
        return _skill_tool(ctx, sid, command, "task_done", {"id": data.get("id", 0)})
    return _skill_tool(ctx, sid, "list", "task_list", {})


def skill_reminders(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.reminders"
    if command == "add":
        return _skill_tool(
            ctx, sid, command, "reminder_add",
            {"title": data.get("title", ""), "at": data.get("at", ""), "minutes": data.get("minutes", 0)},
        )
    return _skill_tool(ctx, sid, "list", "reminder_list", {})


def skill_calendar(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.calendar"
    if command == "add":
        return _skill_tool(
            ctx, sid, command, "event_add",
            {"title": data.get("title", ""), "start": data.get("start", ""), "end": data.get("end", "")},
        )
    return _skill_tool(ctx, sid, "list", "event_list", {})


def skill_notes(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.notes"
    if command == "add":
        return _skill_tool(
            ctx, sid, command, "note_add", {"title": data.get("title", ""), "body": data.get("body", "")}
        )
    return _skill_tool(ctx, sid, "list", "note_list", {})


def skill_focus(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.focus_timer"
    if command == "start":
        return _skill_tool(ctx, sid, command, "focus_start", {"minutes": data.get("minutes", 25)})
    if command == "stop":
        return _skill_tool(ctx, sid, command, "focus_stop", {})
    if ctx.focus_end and ctx.focus_end > time.time():
        remaining = int((ctx.focus_end - time.time()) / 60) + 1
        return _skill_message(sid, "status", f"Focus session active · {remaining} min remaining.")
    return _skill_message(sid, "status", "No focus session running.")


def skill_daily_planner(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.daily_planner"
    tasks = ctx.store.query("SELECT title, done FROM tasks ORDER BY done ASC, id DESC LIMIT 20")
    reminders = ctx.store.query("SELECT title, at FROM reminders WHERE fired=0 ORDER BY at LIMIT 10")
    events = ctx.store.query("SELECT title, start FROM events ORDER BY start LIMIT 10")
    lines = [f"Daily pulse · {datetime.now().strftime('%A %d %B %Y')}", ""]
    lines.append("Pending tasks:")
    pending = [t for t in tasks if not t["done"]]
    lines.extend([f"- {t['title']}" for t in pending] or ["- none"])
    lines.append("\nUpcoming reminders:")
    lines.extend([f"- {r['title']} · {human_ts(r['at'])}" for r in reminders] or ["- none"])
    lines.append("\nCalendar:")
    lines.extend([f"- {e['title']} · {e['start']}" for e in events] or ["- none"])
    if pending:
        lines.append(f"\nSuggested focus: “{pending[0]['title']}” for 25 minutes.")
    text = "\n".join(lines)
    return _skill_message(sid, "digest", text, {"tasks": tasks, "reminders": reminders, "events": events})


def _latest_document(ctx: AppContext, extension: str) -> Optional[Dict[str, Any]]:
    result = ctx.tools.execute("file_search", {"extension": extension, "limit": 5})
    files = (result.data or {}).get("files", [])
    if not files:
        docs = [d for d in ctx.vault.documents() if str(d.get("path", "")).lower().endswith("." + extension)]
        return docs[0] if docs else None
    imported = ctx.tools.execute("vault_import", {"path": files[0]["path"]})
    if not imported.ok:
        return None
    doc_id = (imported.data or {}).get("doc_id")
    return ctx.store.query_one("SELECT * FROM documents WHERE id=?", (doc_id,))


def _document_reader(ctx: AppContext, sid: str, command: str, data: Dict[str, Any], extension: str) -> SkillResult:
    document = str(data.get("document") or data.get("path") or "").strip()
    row = None
    if document:
        row = _find_document(ctx, document)
        if row is None:
            candidate = Path(document)
            if candidate.exists():
                imported = ctx.tools.execute("vault_import", {"path": str(candidate)})
                if not imported.ok:
                    return SkillResult(False, sid, command, {}, imported.error or "Import failed", imported.error)
                doc_id = (imported.data or {}).get("doc_id")
                row = ctx.store.query_one("SELECT * FROM documents WHERE id=?", (doc_id,))
    if row is None:
        row = _latest_document(ctx, extension)
    if row is None:
        return _skill_error(sid, command, f"No .{extension} document is available. Import one in the Vault first.")
    text = ctx.vault.document_text(int(row["id"]))
    excerpt = text[:4000]
    message = f"{row['title']} ({row['pages'] or '?'} pages)\n\n{excerpt}"
    if len(text) > len(excerpt):
        message += "\n\n…truncated. Use Vault search or summarize for more."
    return SkillResult(True, sid, command, {"doc_id": row["id"], "chars": len(text)}, message)


def skill_pdf_reader(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _document_reader(ctx, "boshra.pdf_reader", command, data, "pdf")


def skill_docx_reader(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _document_reader(ctx, "boshra.docx_reader", command, data, "docx")


def skill_summarizer(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    document = str(data.get("document", "")).strip()
    if not document:
        row = _find_document(ctx, "")
        if row is None and ctx.vault.documents():
            row = ctx.vault.documents()[0]
        document = str(row["id"]) if row else ""
    return _skill_tool(
        ctx, "boshra.summarizer", command, "vault_summarize",
        {"document": document, "sentences": data.get("sentences", 6)},
    )


def skill_ocr(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.ocr"
    path = str(data.get("path") or data.get("document") or "")
    if path:
        return _skill_tool(ctx, sid, command, "vault_import", {"path": path})
    images = ctx.tools.execute("file_search", {"extension": "png", "limit": 3})
    files = (images.data or {}).get("files", [])
    if not files:
        return _skill_error(sid, command, "No image found. Provide a path to a scanned document.")
    return _skill_tool(ctx, sid, command, "vault_import", {"path": files[0]["path"]})


def skill_doc_search(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.doc_search", command, "vault_search", {"query": data.get("query", "")})


def skill_report_generator(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.report_generator"
    tasks = ctx.store.query("SELECT title, done FROM tasks LIMIT 100")
    docs = ctx.vault.documents()
    body = ["## Tasks", ""]
    body += [f"- [{'x' if t['done'] else ' '}] {t['title']}" for t in tasks] or ["- none"]
    body += ["", "## Vault documents", ""]
    body += [f"- {d['title']} ({d['pages']} pages)" for d in docs] or ["- none"]
    body += ["", "## Notes", ""]
    body += [f"- {r['title']}" for r in ctx.store.query("SELECT title FROM notes LIMIT 50")] or ["- none"]
    return _skill_tool(
        ctx, sid, command, "report",
        {"title": data.get("title", "Boshra report"), "body": "\n".join(body)},
    )


def skill_app_launcher(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.app_launcher", command, "open_app", {"app": data.get("app", "")})


def skill_file_search(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(
        ctx, "boshra.file_search", command, "file_search",
        {"query": data.get("query", ""), "extension": data.get("extension", ""), "limit": 25},
    )


def skill_file_organizer(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.file_organizer"
    folder = str(data.get("folder", ""))
    if command == "preview":
        preview = tool_organize_preview(ctx, {"folder": folder})
        return SkillResult(preview.ok, sid, command, preview.data, preview.message, preview.error)
    result = tool_organize_apply(ctx, {"folder": folder})
    return SkillResult(result.ok, sid, command, result.data, result.message, result.error)


def skill_diagnostics(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    checks = ctx.diagnostics.run_all() if ctx.diagnostics else []
    return _skill_message(
        "boshra.device_diagnostics", command, DiagnosticsEngine.report_text(checks), {"checks": checks}
    )


def skill_battery(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.battery", command, "battery", {})


def skill_storage(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.storage", command, "storage", {})


def skill_notifications(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.notifications"
    if command == "show":
        return _skill_tool(ctx, sid, command, "notify", {"message": data.get("message", "Boshra notification")})
    return _skill_error(
        sid, command,
        "Reading other apps' notifications requires a native NotificationListenerService with "
        "explicit user activation. It is intentionally not implemented in this build.",
    )


def skill_translator(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(
        ctx, "boshra.translator", command, "translate",
        {"text": data.get("text", ""), "target": data.get("target", "")},
    )


def skill_research(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.research"
    query = str(data.get("query", ""))
    if not query:
        return _skill_error(sid, command, "Provide a research question")
    vault = ctx.tools.execute("vault_answer", {"query": query})
    memory = ctx.tools.execute("memory_search", {"query": query, "k": 3})
    parts = []
    if vault.ok:
        parts.append("From your vault:\n" + vault.render())
    if memory.ok:
        parts.append("From memory:\n" + memory.render())
    if not parts:
        return _skill_error(sid, command, "No local evidence found. Import documents or add memories.")
    return _skill_message(sid, command, "\n\n".join(parts))


def skill_learning_coach(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.learning_coach"
    topic = str(data.get("topic", "")).strip() or "general subject"
    days = max(1, min(_int(data, "days", 7), 90))
    plan = [f"{days}-day learning plan · {topic}", ""]
    for day in range(1, min(days, 14) + 1):
        focus = ["fundamentals", "guided practice", "recall test", "project work", "review"][day % 5]
        plan.append(f"Day {day}: {focus} — 25 min focused study + 5 min recap.")
    if days > 14:
        plan.append(f"…continue the same cycle until day {days}.")
    plan.append("\nRule: end each session by writing three things you can now explain without notes.")
    if data.get("save", True):
        ctx.tools.execute("note_add", {"title": f"Learning plan: {topic}", "body": "\n".join(plan)})
    return _skill_message(sid, command, "\n".join(plan))


def skill_writing(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.writing"
    topic = str(data.get("topic", "")).strip()
    style = str(data.get("style", "clear and concise"))
    if not topic:
        return _skill_error(sid, command, "Provide a topic")
    if ctx.llm.engine_name.startswith("llama"):
        prompt = f"Write a {style} piece about: {topic}"
        chunks = list(ctx.llm.generate_stream(prompt, "You are a precise writing assistant.", 700, 0.7, None))
        text = "".join(chunks).strip()
        if text:
            return _skill_message(sid, command, text, {"model": True})
    outline = (
        f"Draft outline · {topic} ({style})\n\n"
        f"1. Opening: the single most important idea about {topic}.\n"
        "2. Context: why it matters now.\n"
        "3. Three concrete points with a fact or example each.\n"
        "4. Counterpoint and limitation.\n"
        "5. Closing: one actionable takeaway.\n\n"
        "No GGUF model is loaded, so this is a deterministic outline, not generated prose."
    )
    return _skill_message(sid, command, outline, {"model": False})


def skill_knowledge_search(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.knowledge_search", command, "vault_search", {"query": data.get("query", "")})


def skill_memory_search(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    query = str(data.get("query", "")).strip()
    if query:
        return _skill_tool(ctx, "boshra.memory_search", command, "memory_search", {"query": query})
    rows = ctx.memory.list(limit=100)
    text = "\n".join(f"- [{r['category']}] {r['content'][:200]}" for r in rows) or "No memories yet."
    return _skill_message("boshra.memory_search", command, text, {"memories": rows})


def skill_python_assistant(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    sid = "boshra.python_assistant"
    request_text = str(data.get("request", "")).strip()
    if not request_text:
        return _skill_error(sid, command, "Describe the Python task")
    if ctx.llm.engine_name.startswith("llama"):
        prompt = (
            "Write Python 3 code for this request. Explain key decisions briefly after the code. "
            "Do not claim to execute anything.\n\n" + request_text
        )
        chunks = list(ctx.llm.generate_stream(prompt, "You are a senior Python engineer.", 800, 0.4, None))
        text = "".join(chunks).strip()
        if text:
            return _skill_message(sid, command, text, {"model": True})
    return _skill_error(
        sid, command,
        "Python code generation requires a loaded local GGUF model. The deterministic offline "
        "engine can still explain, review and generate tests for code you paste.",
    )


def skill_debugging(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.debugging", command, "debug", {"traceback": data.get("traceback", "")})


def skill_refactoring(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.refactoring", command, "refactor", {"code": data.get("code", "")})


def skill_secure_coding(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.secure_coding", command, "secure_review", {"code": data.get("code", "")})


def skill_code_explainer(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.code_explainer", command, "code_explain", {"code": data.get("code", "")})


def skill_test_generator(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(
        ctx, "boshra.test_generator", command, "test_generate",
        {"code": data.get("code", ""), "module": data.get("module", "module")},
    )


def skill_web_search(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.web_search", command, "web_search", {"query": data.get("query", "")})


def skill_news(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.news", command, "news", {})


def skill_weather(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.weather", command, "weather", {"city": data.get("city", "")})


def skill_rss(ctx: AppContext, command: str, data: Dict[str, Any]) -> SkillResult:
    return _skill_tool(ctx, "boshra.rss", command, "rss", {"url": data.get("url", ""), "limit": 8})


def _bi(
    skill_id: str,
    name: str,
    category: str,
    description: str,
    commands: Sequence[str],
    handler: Callable[[AppContext, str, Dict[str, Any]], SkillResult],
    permissions: Sequence[str] = (),
    risk: str = "low",
    internet: bool = False,
    offline: Optional[bool] = None,
    author: str = "Boshra Core",
) -> Tuple[Dict[str, Any], Callable[[AppContext, str, Dict[str, Any]], SkillResult]]:
    return (
        {
            "id": skill_id,
            "name": name,
            "version": "1.0.0",
            "description": description,
            "author": author,
            "category": category,
            "languages": ["ar", "en"],
            "permissions": list(permissions),
            "optional_permissions": [],
            "dependencies": [],
            "commands": [{"name": c, "description": ""} for c in commands],
            "offline": (not internet) if offline is None else offline,
            "internet_required": internet,
            "risk_level": risk,
            "min_app_version": "4.0.0",
            "config_schema": {},
        },
        handler,
    )


BUILTIN_SKILLS: List[Tuple[Dict[str, Any], Callable[[AppContext, str, Dict[str, Any]], SkillResult]]] = [
    # productivity
    _bi("boshra.task_manager", "Task Manager", "productivity",
        "Create, list and complete local tasks", ["add", "list", "done"], skill_task_manager),
    _bi("boshra.reminders", "Reminder Manager", "productivity",
        "Local reminders with device notifications", ["add", "list"], skill_reminders,
        permissions=("notifications.show",)),
    _bi("boshra.calendar", "Calendar Assistant", "productivity",
        "Local calendar entries stored on-device (no system calendar sync)", ["add", "list"], skill_calendar),
    _bi("boshra.notes", "Notes Manager", "productivity",
        "Encrypted local notes", ["add", "list"], skill_notes),
    _bi("boshra.focus_timer", "Focus Timer", "productivity",
        "Start and stop a focus session", ["start", "stop", "status"], skill_focus),
    _bi("boshra.daily_planner", "Daily Planner", "productivity",
        "Aggregate tasks, reminders and calendar into a daily pulse", ["digest"], skill_daily_planner),
    # documents
    _bi("boshra.pdf_reader", "PDF Reader", "documents",
        "Read and index PDF documents locally", ["read"], skill_pdf_reader, ("documents.read",)),
    _bi("boshra.docx_reader", "DOCX Reader", "documents",
        "Extract text from Word documents", ["read"], skill_docx_reader, ("documents.read",)),
    _bi("boshra.summarizer", "Document Summarizer", "documents",
        "Extractive summaries with document provenance", ["summarize"], skill_summarizer, ("documents.read",)),
    _bi("boshra.ocr", "OCR Extractor", "documents",
        "Extract text from scanned images when an OCR engine is installed", ["extract"], skill_ocr,
        ("documents.read",)),
    _bi("boshra.doc_search", "Document Search", "documents",
        "Semantic and keyword search across indexed passages", ["search"], skill_doc_search,
        ("documents.read",)),
    _bi("boshra.report_generator", "Report Generator", "documents",
        "Generate a markdown report from your local data", ["generate"], skill_report_generator,
        ("documents.write",), "medium"),
    # system
    _bi("boshra.app_launcher", "App Launcher", "system",
        "Open installed Android applications through intents", ["launch"], skill_app_launcher,
        ("android.app.open",), "medium"),
    _bi("boshra.file_search", "File Search", "system",
        "Search approved Boshra folders by name or extension", ["search"], skill_file_search,
        ("files.search",)),
    _bi("boshra.file_organizer", "File Organizer", "system",
        "Sort files into categories inside the Boshra workspace; preview first", ["preview", "apply"],
        skill_file_organizer, ("files.move",), "high"),
    _bi("boshra.device_diagnostics", "Device Diagnostics", "system",
        "Check engine, storage, permissions and integrity", ["run"], skill_diagnostics, ("device.info",)),
    _bi("boshra.battery", "Battery Information", "system",
        "Battery percentage and charging state", ["status"], skill_battery, ("device.info",)),
    _bi("boshra.storage", "Storage Information", "system",
        "Free and total local storage", ["status"], skill_storage, ("device.info",)),
    _bi("boshra.notifications", "Notification Reader", "system",
        "Show Boshra notifications; reading other apps requires a native listener", ["read", "show"],
        skill_notifications, ("notifications.read",), "high"),
    # intelligence
    _bi("boshra.translator", "Translator", "intelligence",
        "Bilingual AR/EN translation using a local dictionary or local model", ["translate"],
        skill_translator),
    _bi("boshra.research", "Research Assistant", "intelligence",
        "Combine vault evidence and memory for a research question", ["research"], skill_research,
        ("documents.read", "memory.read")),
    _bi("boshra.learning_coach", "Learning Coach", "intelligence",
        "Generate a structured local study plan", ["plan"], skill_learning_coach),
    _bi("boshra.writing", "Writing Assistant", "intelligence",
        "Draft text with the local model, or produce a deterministic outline", ["write"], skill_writing),
    _bi("boshra.knowledge_search", "Knowledge Search", "intelligence",
        "Search indexed knowledge passages with sources", ["search"], skill_knowledge_search,
        ("documents.read",)),
    _bi("boshra.memory_search", "Personal Memory Search", "intelligence",
        "Search everything you asked Boshra to remember", ["search", "list"], skill_memory_search,
        ("memory.read",)),
    # development
    _bi("boshra.python_assistant", "Python Assistant", "development",
        "Generate Python with a loaded GGUF model", ["generate"], skill_python_assistant),
    _bi("boshra.debugging", "Debugging Assistant", "development",
        "Analyse a traceback and suggest causes", ["analyse"], skill_debugging),
    _bi("boshra.refactoring", "Refactoring Assistant", "development",
        "AST-based refactoring suggestions", ["review"], skill_refactoring),
    _bi("boshra.secure_coding", "Secure Coding Assistant", "development",
        "Pattern-based security review", ["review"], skill_secure_coding),
    _bi("boshra.code_explainer", "Code Explainer", "development",
        "Explain Python structure without executing it", ["explain"], skill_code_explainer),
    _bi("boshra.test_generator", "Test Generator", "development",
        "Generate pytest skeletons from real signatures", ["generate"], skill_test_generator),
    # online
    _bi("boshra.web_search", "Web Search", "online",
        "Search the web when the user explicitly asks", ["search"], skill_web_search,
        ("network.access",), "medium", internet=True),
    _bi("boshra.news", "News Reader", "online",
        "Fetch RSS headlines", ["read"], skill_news, ("network.access",), "medium", internet=True),
    _bi("boshra.weather", "Weather Assistant", "online",
        "Current weather for a city", ["current"], skill_weather,
        ("network.access",), "medium", internet=True),
    _bi("boshra.rss", "RSS Reader", "online",
        "Read a specific RSS feed URL", ["read"], skill_rss,
        ("network.access",), "medium", internet=True),
]


# --------------------------------------------------------------------------
# External skill package execution (isolated subprocess, not a sandbox)
# --------------------------------------------------------------------------
def run_skill_file_cli(skill_dir: str, command: str, data_json: str) -> int:
    """Entry point for `main.py --run-skill ...`. Prints one JSON object."""
    import importlib.util

    try:
        directory = Path(skill_dir).resolve()
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = str(manifest.get("entry_point", "skill.py"))
        entry_path = (directory / entry).resolve()
        if not is_within(entry_path, directory) or not entry_path.is_file():
            raise RuntimeError("invalid entry point")
        data = json.loads(data_json or "{}")
        spec = importlib.util.spec_from_file_location("boshra_external_skill", str(entry_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load entry point")
        module = importlib.util.module_from_spec(spec)
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            spec.loader.exec_module(module)

            class _Context:
                offline = True
                internet_required = bool(manifest.get("internet_required", False))

                @staticmethod
                def log(_message: str) -> None:
                    return

            result = None
            if hasattr(module, "execute"):
                result = module.execute(command, data)
            elif hasattr(module, "Skill"):
                instance = module.Skill()
                if hasattr(instance, "initialize"):
                    instance.initialize(_Context())
                result = instance.execute(command, data)
            elif hasattr(module, command):
                result = getattr(module, command)(data)
            else:
                raise RuntimeError("skill.py exposes neither execute() nor a Skill class")
        payload = {"ok": True, "result": result, "stdout": captured.getvalue()[-2000:]}
    except Exception as exc:  # noqa: BLE001
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str))
    return 0


class SkillEngine:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store
        self.config = ctx.config
        self._lock = threading.RLock()
        self._builtin_handlers: Dict[str, Callable[[AppContext, str, Dict[str, Any]], SkillResult]] = {}
        self._builtin_manifests: Dict[str, Dict[str, Any]] = {}
        self.invalid: List[Dict[str, str]] = []
        self._register_builtins()
        self.discover()

    # -- registration ----------------------------------------------------
    def _known_tools(self) -> List[str]:
        return self.ctx.tools.names() if self.ctx.tools else []

    def _upsert(
        self, manifest: Dict[str, Any], source: str, path: str = "", enabled: Optional[bool] = None
    ) -> None:
        existing = self.store.query_one("SELECT id, enabled FROM skills WHERE id=?", (manifest["id"],))
        payload = json.dumps(manifest, ensure_ascii=False)
        now = now_iso()
        if existing:
            self.store.execute(
                "UPDATE skills SET name=?, version=?, category=?, description=?, risk=?, offline=?, "
                "source=?, path=?, manifest=?, health='healthy' WHERE id=?",
                (
                    manifest["name"], manifest["version"], manifest["category"],
                    manifest["description"], manifest["risk_level"], 1 if manifest["offline"] else 0,
                    source, path, payload, manifest["id"],
                ),
            )
            if enabled is not None:
                self.store.execute("UPDATE skills SET enabled=? WHERE id=?", (1 if enabled else 0, manifest["id"]))
        else:
            self.store.execute(
                "INSERT INTO skills(id,name,version,category,description,risk,offline,source,path,manifest,"
                "enabled,health,last_run,errors,installed) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    manifest["id"], manifest["name"], manifest["version"], manifest["category"],
                    manifest["description"], manifest["risk_level"], 1 if manifest["offline"] else 0,
                    source, path, payload, 1 if (enabled if enabled is not None else True) else 0,
                    "healthy", now, 0, now,
                ),
            )

    def _register_builtins(self) -> None:
        for raw, handler in BUILTIN_SKILLS:
            manifest, errors = validate_manifest(raw, known_tools=self._known_tools(), builtin=True)
            if errors or manifest is None:
                LOG.error(f"builtin skill invalid {raw.get('id')}: {errors}")
                continue
            self._builtin_handlers[manifest["id"]] = handler
            self._builtin_manifests[manifest["id"]] = manifest
            self._upsert(manifest, "builtin", "")
            for command in manifest["commands"]:
                self.store.execute(
                    "INSERT INTO skill_history(skill_id, action, success, message, ts) VALUES(?,?,1,?,?)",
                    (manifest["id"], "registered", "built-in available", now_iso()),
                )

    # -- discovery -------------------------------------------------------
    def discover(self) -> Dict[str, int]:
        registered = 0
        invalid = 0
        self.invalid = []
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        for folder in sorted(SKILLS_DIR.iterdir()):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            manifest_path = folder / "manifest.json"
            if not manifest_path.is_file():
                self.invalid.append({"path": str(folder), "error": "manifest.json missing"})
                invalid += 1
                continue
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self.invalid.append({"path": str(folder), "error": f"Malformed JSON: {exc}"})
                invalid += 1
                continue
            manifest, errors = validate_manifest(raw, skill_dir=folder, known_tools=self._known_tools())
            if errors or manifest is None:
                skill_id = str(raw.get("id", "")) if isinstance(raw, dict) else ""
                if skill_id and SKILL_ID_RE.match(skill_id):
                    self.store.execute(
                        "INSERT INTO skills(id,name,version,category,description,risk,offline,source,path,"
                        "manifest,enabled,health,last_run,errors,installed) VALUES(?,?,?,?,?,?,?,?,?,?,0,?,?,?,?) "
                        "ON CONFLICT(id) DO UPDATE SET health='invalid', path=excluded.path, manifest=excluded.manifest",
                        (
                            skill_id, str(raw.get("name", skill_id))[:80], str(raw.get("version", "0.0.0")),
                            str(raw.get("category", "custom")), str(raw.get("description", ""))[:400],
                            str(raw.get("risk_level", "low")), 1, "external", str(folder),
                            json.dumps(raw, ensure_ascii=False), "invalid", "", 1, now_iso(),
                        ),
                    )
                self.invalid.append({"path": str(folder), "error": "; ".join(errors) or "invalid manifest"})
                self.store.audit("skill", "invalid-manifest", f"{folder.name}: {'; '.join(errors)[:500]}", "medium")
                invalid += 1
                continue
            existing = self.store.query_one("SELECT version, source FROM skills WHERE id=?", (manifest["id"],))
            if existing and existing["source"] == "builtin":
                self.invalid.append({"path": str(folder), "error": "Cannot override a built-in skill id"})
                invalid += 1
                continue
            if existing and _version_tuple(existing["version"]) > _version_tuple(manifest["version"]):
                self.store.audit("skill", "older-version-ignored", manifest["id"], "low")
                continue
            self._upsert(manifest, "external", str(folder), enabled=False if not existing else None)
            self.store.audit("skill", "discovered", f"{manifest['id']} v{manifest['version']}", "low")
            registered += 1
        return {"registered": registered, "invalid": invalid}

    # -- queries ---------------------------------------------------------
    def list_skills(self, query: str = "", category: str = "") -> List[Dict[str, Any]]:
        rows = self.store.query("SELECT * FROM skills ORDER BY category, name")
        result: List[Dict[str, Any]] = []
        needle = normalize_arabic(query.lower()) if query else ""
        for row in rows:
            manifest = {}
            with contextlib.suppress(ValueError):
                manifest = json.loads(row["manifest"])
            permissions = manifest.get("permissions", [])
            haystack = normalize_arabic(
                f"{row['id']} {row['name']} {row['description']} {row['category']} "
                f"{' '.join(permissions)} {' '.join(c.get('name', '') for c in manifest.get('commands', []))}".lower()
            )
            if needle and needle not in haystack:
                continue
            if category and row["category"] != category:
                continue
            result.append(
                {
                    **row,
                    "enabled": bool(row["enabled"]),
                    "offline": bool(row["offline"]),
                    "manifest": manifest,
                    "permissions": permissions,
                    "optional_permissions": manifest.get("optional_permissions", []),
                    "commands": manifest.get("commands", []),
                    "languages": manifest.get("languages", []),
                    "internet_required": manifest.get("internet_required", False),
                    "builtin": row["source"] == "builtin",
                }
            )
        return result

    def get(self, skill_id: str) -> Optional[Dict[str, Any]]:
        for skill in self.list_skills():
            if skill["id"] == skill_id:
                return skill
        return None

    def grants(self, skill_id: str) -> Dict[str, bool]:
        rows = self.store.query("SELECT permission, granted FROM skill_grants WHERE skill_id=?", (skill_id,))
        return {row["permission"]: bool(row["granted"]) for row in rows}

    def grant(self, skill_id: str, permission: str) -> bool:
        skill = self.get(skill_id)
        if not skill or permission not in skill["permissions"] + skill["optional_permissions"]:
            return False
        if not self.ctx.permissions.request(permission, f"Skill {skill['name']} requires it"):
            return False
        self.store.execute(
            "INSERT INTO skill_grants(skill_id, permission, granted, ts) VALUES(?,?,1,?) "
            "ON CONFLICT(skill_id,permission) DO UPDATE SET granted=1, ts=excluded.ts",
            (skill_id, permission, now_iso()),
        )
        self.store.audit("skill", "grant", f"{skill_id}:{permission}", PERMISSION_CATALOG.get(permission, ("low",))[0])
        return True

    def revoke(self, skill_id: str, permission: str) -> bool:
        self.store.execute(
            "INSERT INTO skill_grants(skill_id, permission, granted, ts) VALUES(?,?,0,?) "
            "ON CONFLICT(skill_id,permission) DO UPDATE SET granted=0, ts=excluded.ts",
            (skill_id, permission, now_iso()),
        )
        self.store.audit("skill", "revoke", f"{skill_id}:{permission}", "medium")
        return True

    def enable(self, skill_id: str) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill:
            return {"ok": False, "error": "Skill not found"}
        if skill["health"] == "invalid":
            return {"ok": False, "error": "Skill manifest is invalid; reinstall a valid version"}
        if skill["internet_required"] and self.config.online_mode == "local":
            return {"ok": False, "error": "This skill needs internet but Online mode is LOCAL ONLY"}
        if not skill["builtin"]:
            missing = [p for p in skill["permissions"] if not self.grants(skill_id).get(p)]
            if missing:
                return {"ok": False, "error": "Grant required permissions first: " + ", ".join(missing)}
        self.store.execute("UPDATE skills SET enabled=1, health='healthy' WHERE id=?", (skill_id,))
        self.store.audit("skill", "enable", skill_id, skill["risk"])
        return {"ok": True, "message": f"{skill['name']} enabled"}

    def disable(self, skill_id: str) -> Dict[str, Any]:
        self.store.execute("UPDATE skills SET enabled=0 WHERE id=?", (skill_id,))
        self.store.audit("skill", "disable", skill_id, "low")
        return {"ok": True, "message": "Skill disabled"}

    def disable_unhealthy(self) -> int:
        rows = self.store.query("SELECT id FROM skills WHERE health NOT IN ('healthy','disabled')")
        for row in rows:
            self.store.execute("UPDATE skills SET enabled=0 WHERE id=?", (row["id"],))
        return len(rows)

    def uninstall(self, skill_id: str) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill:
            return {"ok": False, "error": "Skill not found"}
        if skill["builtin"]:
            return {"ok": False, "error": "Built-in skills cannot be uninstalled; disable it instead"}
        path = Path(skill["path"]) if skill["path"] else None
        if path and path.exists() and is_within(path, SKILLS_DIR):
            with contextlib.suppress(OSError):
                shutil.rmtree(path)
        self.store.execute("DELETE FROM skill_grants WHERE skill_id=?", (skill_id,))
        self.store.execute("DELETE FROM skills WHERE id=?", (skill_id,))
        self.store.audit("skill", "uninstall", skill_id, "high")
        return {"ok": True, "message": f"Uninstalled {skill['name']}"}

    def history(self, skill_id: str, limit: int = 40) -> List[Dict[str, Any]]:
        return self.store.query(
            "SELECT * FROM skill_history WHERE skill_id=? ORDER BY id DESC LIMIT ?", (skill_id, limit)
        )

    def reset_config(self, skill_id: str) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill:
            return {"ok": False, "error": "Skill not found"}
        path = Path(skill["path"]) / "config.json" if skill["path"] else None
        if path and path.exists():
            with contextlib.suppress(OSError):
                path.unlink()
        self.store.audit("skill", "reset-config", skill_id, "low")
        return {"ok": True, "message": "Skill configuration reset"}

    # -- execution -------------------------------------------------------
    def _record(self, skill_id: str, action: str, ok: bool, message: str) -> None:
        self.store.execute(
            "INSERT INTO skill_history(skill_id, action, success, message, ts) VALUES(?,?,?,?,?)",
            (skill_id, action, 1 if ok else 0, (message or "")[:1500], now_iso()),
        )
        if ok:
            self.store.execute(
                "UPDATE skills SET last_run=?, health='healthy' WHERE id=?", (now_iso(), skill_id)
            )
        else:
            self.store.execute(
                "UPDATE skills SET errors=errors+1, health=CASE WHEN errors>=2 THEN 'broken' ELSE 'caution' END "
                "WHERE id=?",
                (skill_id,),
            )

    def _permission_gate(self, skill: Dict[str, Any]) -> Optional[str]:
        if skill["builtin"]:
            return None
        granted = self.grants(skill["id"])
        missing = [p for p in skill["permissions"] if not granted.get(p)]
        if missing:
            return "Missing skill permissions: " + ", ".join(missing)
        return None

    def _render(self, value: Any, data: Dict[str, Any], last: str) -> Any:
        if isinstance(value, str):
            def replace(match: re.Match[str]) -> str:
                key = match.group(1).strip()
                if key == "last":
                    return last
                if key.startswith("input."):
                    key = key[6:]
                return str(data.get(key, ""))

            return re.sub(r"\{\{\s*([^}]+)\s*\}\}", replace, value)
        if isinstance(value, list):
            return [self._render(item, data, last) for item in value]
        if isinstance(value, dict):
            return {k: self._render(v, data, last) for k, v in value.items()}
        return value

    def _execute_steps(
        self, skill: Dict[str, Any], command: str, steps: Sequence[Dict[str, Any]], data: Dict[str, Any]
    ) -> SkillResult:
        last = ""
        results: List[Any] = []
        for index, step in enumerate(steps, start=1):
            tool = step.get("tool", "")
            spec = self.ctx.tools.get(tool)
            if spec is None:
                return _skill_error(
                    skill["id"], command, f"Step {index}: tool '{tool}' is not registered"
                )
            args = self._render(step.get("args", {}), data, last)
            if not isinstance(args, dict):
                args = {}
            result = self.ctx.tools.execute(tool, args)
            if result.needs_confirmation:
                approved = self.ctx.confirm(
                    f"Skill {skill['name']} wants to run a destructive step.\n\n{result.message}",
                    result.message,
                )
                if not approved:
                    return _skill_error(skill["id"], command, f"Step {index} cancelled by user")
                result = self.ctx.tools.execute(tool, {**args, "_confirmed": True})
            results.append({"step": index, "tool": tool, "ok": result.ok, "message": result.render()})
            if not result.ok:
                return SkillResult(False, skill["id"], command, {"steps": results}, result.render(), result.error)
            last = result.message or last
            self.ctx.store.audit("skill", f"{skill['id']}:{tool}", result.message[:400], skill["risk"])
        message = last or "Skill completed."
        return SkillResult(True, skill["id"], command, {"steps": results}, message)

    def _execute_python(self, skill: Dict[str, Any], command: str, data: Dict[str, Any]) -> SkillResult:
        if not self.config.allow_python_skills:
            return _skill_error(
                skill["id"], command,
                "Python skills are disabled in Settings. Only enable this for code you have reviewed: "
                "a Python subprocess is isolated from the UI but is NOT a security sandbox.",
            )
        if IS_ANDROID and not shutil.which("python3"):
            return _skill_error(
                skill["id"], command,
                "Spawning a Python subprocess is not available in this Android build. "
                "Use a declarative skill (tool steps) instead.",
            )
        python = sys.executable if "python" in Path(sys.executable).name else "python3"
        if not python:
            return _skill_error(skill["id"], command, "No Python interpreter available for subprocess isolation")
        directory = Path(skill["path"])
        if not is_within(directory, SKILLS_DIR):
            return _skill_error(skill["id"], command, "Skill path is outside the skills directory")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(PRIVATE_DIR),
            "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            process = subprocess.run(
                [python, str(Path(__file__).resolve()), "--run-skill", str(directory), command,
                 json.dumps(data, ensure_ascii=False)],
                capture_output=True, text=True, timeout=30,
                cwd=str(directory), env=env,
            )
            stdout = process.stdout.strip()
            if not stdout:
                return _skill_error(
                    skill["id"], command,
                    f"Skill process produced no output (exit {process.returncode}): {process.stderr[-400:]}",
                )
            payload = json.loads(stdout.splitlines()[-1])
            if not payload.get("ok"):
                return _skill_error(skill["id"], command, str(payload.get("error", "Skill failed")))
            return SkillResult(
                True, skill["id"], command, payload.get("result"),
                "Skill executed in an isolated subprocess (not a sandbox).",
            )
        except subprocess.TimeoutExpired:
            return _skill_error(skill["id"], command, "Skill timed out after 30 seconds")
        except (OSError, ValueError) as exc:
            return _skill_error(skill["id"], command, f"Skill process error: {exc}")

    def execute(self, skill_id: str, command: str = "", data: Optional[Dict[str, Any]] = None) -> SkillResult:
        data = dict(data or {})
        skill = self.get(skill_id)
        if not skill:
            return _skill_error(skill_id, command, "Skill not found")
        if not skill["enabled"]:
            return _skill_error(skill_id, command, "Skill is disabled and cannot execute")
        commands = [c["name"] for c in skill["commands"]]
        command = command or (commands[0] if commands else "run")
        if commands and command not in commands:
            return _skill_error(skill_id, command, f"Unknown command. Available: {', '.join(commands)}")
        gate_error = self._permission_gate(skill)
        if gate_error:
            return _skill_error(skill_id, command, gate_error)
        if skill["internet_required"]:
            network = self.ctx.bridge.network_info()
            if not network.get("online"):
                return _skill_error(skill_id, command, "This skill requires internet, which is unavailable")
            if self.config.online_mode == "local":
                return _skill_error(skill_id, command, "Online mode is LOCAL ONLY")
        if skill["risk"] == "high" and not data.get("_confirmed"):
            if not self.ctx.confirm(
                f"High-risk skill: {skill['name']}\n\nCommand: {command}\n\n"
                f"{json.dumps(data, ensure_ascii=False)[:400]}\n\nProceed?",
                "High-risk skill execution",
            ):
                return _skill_error(skill_id, command, "Cancelled by user")

        with self._lock:
            self.store.audit("skill", f"execute:{command}", skill_id, skill["risk"])
            try:
                if skill["builtin"]:
                    handler = self._builtin_handlers.get(skill_id)
                    if handler is None:
                        return _skill_error(skill_id, command, "Built-in handler missing")
                    result = handler(self.ctx, command, data)
                elif skill["manifest"].get("entry_point"):
                    result = self._execute_python(skill, command, data)
                else:
                    command_entry = next(
                        (c for c in skill["commands"] if c["name"] == command), None
                    )
                    steps = (command_entry or {}).get("steps") or skill["manifest"].get("steps", [])
                    if not steps:
                        return _skill_error(skill_id, command, "No steps defined for this command")
                    result = self._execute_steps(skill, command, steps, data)
            except Exception as exc:  # noqa: BLE001
                LOG.error(f"skill {skill_id} crashed", exc)
                result = _skill_error(skill_id, command, f"{type(exc).__name__}: {exc}")
            self._record(skill_id, command, result.success, result.message or result.error or "")
            if not result.success:
                self.store.audit("skill", f"failure:{command}", result.error or "", skill["risk"])
            return result

    def self_test(self, skill_id: str) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill:
            return {"ok": False, "error": "Skill not found"}
        checks: List[Dict[str, str]] = []
        checks.append({"name": "manifest", "status": "ok" if skill["health"] != "invalid" else "fail"})
        checks.append({"name": "enabled", "status": "ok" if skill["enabled"] else "warn"})
        if not skill["builtin"]:
            missing = [p for p in skill["permissions"] if not self.grants(skill_id).get(p)]
            checks.append(
                {"name": "permissions", "status": "ok" if not missing else "fail",
                 "detail": ", ".join(missing)}
            )
        if skill["manifest"].get("entry_point"):
            entry = Path(skill["path"]) / skill["manifest"]["entry_point"]
            checks.append({"name": "entry point", "status": "ok" if entry.is_file() else "fail"})
        for step in skill["manifest"].get("steps", []):
            known = self.ctx.tools.get(step["tool"]) is not None
            checks.append(
                {"name": f"tool:{step['tool']}", "status": "ok" if known else "fail"}
            )
        return {
            "ok": all(c["status"] != "fail" for c in checks),
            "checks": checks,
            "message": "Structural self-test only; live execution is logged in skill history.",
        }

    # -- packaging -------------------------------------------------------
    def _zip_skill(self, folder: Path, destination: Path) -> None:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in folder.rglob("*"):
                if path.is_file() and not path.name.startswith("."):
                    archive.write(path, path.relative_to(folder).as_posix())

    def export_skill(self, skill_id: str) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill:
            return {"ok": False, "error": "Skill not found"}
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", skill_id)
        destination = EXPORT_DIR / f"{safe}_{skill['version']}.boshra-skill"
        try:
            if skill["builtin"]:
                temp = Path(tempfile.mkdtemp(prefix="boshra-export-"))
                (temp / "manifest.json").write_text(
                    json.dumps(skill["manifest"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (temp / "README.md").write_text(
                    f"# {skill['name']}\n\nBuilt-in skill manifest exported from {APP_NAME} {APP_VERSION}.\n",
                    encoding="utf-8",
                )
                self._zip_skill(temp, destination)
                shutil.rmtree(temp, ignore_errors=True)
            else:
                folder = Path(skill["path"])
                if not folder.exists():
                    return {"ok": False, "error": "Skill folder missing"}
                self._zip_skill(folder, destination)
        except (OSError, zipfile.BadZipFile) as exc:
            return {"ok": False, "error": f"Export failed: {exc}"}
        self.store.audit("skill", "export", skill_id, "low")
        return {"ok": True, "path": str(destination), "message": f"Exported to {destination.name}"}

    def inspect_package(self, package_path: str) -> Dict[str, Any]:
        path = Path(package_path)
        if not path.exists():
            return {"ok": False, "error": "Package not found"}
        if path.stat().st_size > 20 * 1024 * 1024:
            return {"ok": False, "error": "Package exceeds 20 MB"}
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if len(names) > 200:
                    return {"ok": False, "error": "Package has too many files"}
                for name in names:
                    candidate = Path(name)
                    if candidate.is_absolute() or ".." in candidate.parts:
                        return {"ok": False, "error": f"Unsafe path in package: {name}"}
                manifests = [n for n in names if n.endswith("manifest.json")]
                if not manifests:
                    return {"ok": False, "error": "No manifest.json in package"}
                root = str(Path(manifests[0]).parent)
                manifest_raw = json.loads(archive.read(manifests[0]).decode("utf-8"))
                manifest, errors = validate_manifest(manifest_raw, known_tools=self._known_tools())
                if errors or manifest is None:
                    return {"ok": False, "error": "; ".join(errors)}
                has_python = any(n.endswith(".py") for n in names)
                return {
                    "ok": True,
                    "manifest": manifest,
                    "root": "" if root == "." else root,
                    "has_python": has_python,
                    "files": len(names),
                    "package": str(path),
                }
        except (OSError, zipfile.BadZipFile, ValueError, KeyError) as exc:
            return {"ok": False, "error": f"Invalid package: {exc}"}

    def install_package(self, package_path: str) -> Dict[str, Any]:
        inspection = self.inspect_package(package_path)
        if not inspection.get("ok"):
            return inspection
        manifest = inspection["manifest"]
        if manifest["id"] in self._builtin_manifests:
            return {"ok": False, "error": "Package id collides with a built-in skill"}
        temp = Path(tempfile.mkdtemp(prefix="boshra-skill-"))
        try:
            with zipfile.ZipFile(package_path) as archive:
                archive.extractall(temp)
            root = temp / inspection["root"] if inspection["root"] else temp
            target = SKILLS_DIR / re.sub(r"[^a-z0-9_.-]+", "_", manifest["id"])
            backup = None
            if target.exists():
                backups = SKILLS_DIR / ".backups"
                backups.mkdir(exist_ok=True)
                backup = backups / f"{target.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                shutil.copytree(target, backup)
                shutil.rmtree(target)
                self.store.audit("skill", "rollback-point", str(backup), "low")
            shutil.copytree(root, target)
            # validate against the final location
            manifest_final, errors = validate_manifest(
                json.loads((target / "manifest.json").read_text(encoding="utf-8")),
                skill_dir=target, known_tools=self._known_tools(),
            )
            if errors or manifest_final is None:
                shutil.rmtree(target, ignore_errors=True)
                if backup and backup.exists():
                    shutil.copytree(backup, target)
                return {"ok": False, "error": "; ".join(errors)}
            self._upsert(manifest_final, "external", str(target), enabled=False)
            self.store.audit("skill", "install", manifest_final["id"], manifest_final["risk_level"])
            return {
                "ok": True,
                "skill_id": manifest_final["id"],
                "message": f"Installed {manifest_final['name']} v{manifest_final['version']} (disabled until enabled)",
                "python_entry": bool(manifest_final.get("entry_point")),
                "backup": str(backup) if backup else "",
            }
        except (OSError, zipfile.BadZipFile) as exc:
            return {"ok": False, "error": f"Install failed: {exc}"}
        finally:
            shutil.rmtree(temp, ignore_errors=True)

    def rollback(self, skill_id: str) -> Dict[str, Any]:
        skill = self.get(skill_id)
        if not skill or skill["builtin"]:
            return {"ok": False, "error": "Only external skills can be rolled back"}
        backups = sorted((SKILLS_DIR / ".backups").glob(f"{skill_id.split('.')[-1]}_*"), reverse=True)
        if not backups:
            parent = sorted((SKILLS_DIR / ".backups").glob("*"), reverse=True)
            backups = [b for b in parent if b.is_dir() and skill_id in b.name]
        if not backups:
            return {"ok": False, "error": "No rollback snapshot available"}
        folder = Path(skill["path"])
        try:
            if folder.exists():
                shutil.rmtree(folder)
            shutil.copytree(backups[0], folder)
            self.discover()
            self.store.audit("skill", "rollback", skill_id, "medium")
            return {"ok": True, "message": f"Rolled back {skill_id} from {backups[0].name}"}
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

    # -- skill builder ---------------------------------------------------
    def propose_skill(self, description: str) -> Dict[str, Any]:
        text = (description or "").strip()
        if len(text) < 8:
            return {"ok": False, "error": "Describe the capability in a sentence"}
        lowered = text.lower()
        slug = "user.custom_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        name = text[:48].rstrip(" .") or "Custom skill"
        permissions: List[str] = []
        risk = "low"
        steps: List[Dict[str, Any]] = []
        commands = [{"name": "run", "description": "Run the generated composition"}]
        if any(word in lowered for word in ("summar", "لخص", "تلخيص")) and any(
            word in lowered for word in ("doc", "pdf", "ملف", "مستند")
        ):
            slug = "user.doc_summarizer_" + hashlib.md5(text.encode()).hexdigest()[:6]
            name = "Document summarizer (generated)"
            permissions = ["documents.read", "documents.write"]
            risk = "medium"
            steps = [
                {"tool": "vault_summarize", "args": {"document": "{{document}}", "sentences": 6}},
                {"tool": "report", "args": {"title": "Document summary", "body": "{{last}}"}},
            ]
            commands = [{"name": "summarize", "description": "Summarize a document and export a report"}]
        elif any(word in lowered for word in ("organize", "arrange", "رتب", "نظم")):
            slug = "user.file_organizer_" + hashlib.md5(text.encode()).hexdigest()[:6]
            name = "Workspace organizer (generated)"
            permissions = ["files.search", "files.move"]
            risk = "high"
            steps = [
                {"tool": "file_search", "args": {"extension": "", "limit": 5}},
                {"tool": "organize_files", "args": {"folder": "{{folder}}"}},
            ]
        elif any(word in lowered for word in ("briefing", "digest", "daily", "موجز", "يومي")):
            slug = "user.daily_digest_" + hashlib.md5(text.encode()).hexdigest()[:6]
            name = "Daily digest (generated)"
            steps = [
                {"tool": "task_list", "args": {}},
                {"tool": "reminder_list", "args": {}},
                {"tool": "note_list", "args": {}},
            ]
        else:
            permissions = ["documents.read", "memory.read"]
            steps = [
                {"tool": "vault_answer", "args": {"query": "{{query}}"}},
                {"tool": "memory_search", "args": {"query": "{{query}}", "k": 3}},
            ]
        manifest, errors = validate_manifest(
            {
                "id": slug,
                "name": name,
                "version": "1.0.0",
                "description": text[:400],
                "author": "Skill Builder (local)",
                "category": "custom",
                "languages": ["ar", "en"],
                "permissions": permissions,
                "optional_permissions": [],
                "dependencies": [],
                "commands": commands,
                "steps": steps,
                "min_app_version": "4.0.0",
                "offline": True,
                "internet_required": False,
                "risk_level": risk,
                "config_schema": {},
            },
            known_tools=self._known_tools(),
        )
        if errors or manifest is None:
            return {"ok": False, "error": "; ".join(errors)}
        tests = [
            {"name": "structural-manifest-validation", "status": "passed"},
            {"name": "tool-allowlist-check", "status": "passed"},
            {"name": "dry-run-policy", "status": "preview required for destructive steps" if risk == "high" else "not required"},
        ]
        return {"ok": True, "manifest": manifest, "steps": steps, "tests": tests}

    def install_generated(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        manifest, errors = validate_manifest(manifest, known_tools=self._known_tools())
        if errors or manifest is None:
            return {"ok": False, "error": "; ".join(errors)}
        if manifest["id"] in self._builtin_manifests:
            return {"ok": False, "error": "Generated id collides with a built-in skill"}
        folder = SKILLS_DIR / re.sub(r"[^a-z0-9_.-]+", "_", manifest["id"])
        try:
            folder.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                folder / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
                0o600,
            )
            atomic_write_text(
                folder / "README.md",
                f"# {manifest['name']}\n\nGenerated locally by {APP_NAME} Skill Builder.\n\n"
                f"{manifest['description']}\n",
                0o600,
            )
        except OSError as exc:
            return {"ok": False, "error": f"Could not write skill: {exc}"}
        self._upsert(manifest, "external", str(folder), enabled=True)
        for permission in manifest["permissions"]:
            self.grant(manifest["id"], permission)
        self.store.audit("skill", "install-generated", manifest["id"], manifest["risk_level"])
        return {"ok": True, "skill_id": manifest["id"], "message": f"Installed {manifest['name']}"}


# --------------------------------------------------------------------------
# Automation engine (workflows + recorded routines)
# --------------------------------------------------------------------------
PRESET_ROUTINES: List[Tuple[str, List[Dict[str, Any]]]] = [
    (
        "Morning briefing",
        [
            {"tool": "time_date", "args": {}},
            {"tool": "task_list", "args": {}},
            {"tool": "reminder_list", "args": {}},
        ],
    ),
    (
        "Programming session",
        [
            {"tool": "task_list", "args": {}},
            {"tool": "note_list", "args": {}},
            {"tool": "focus_start", "args": {"minutes": 25}},
        ],
    ),
    (
        "Document review",
        [
            {"tool": "vault_list", "args": {}},
            {"tool": "vault_summarize", "args": {"document": ""}},
        ],
    ),
    (
        "Device check",
        [
            {"tool": "diagnostics", "args": {}},
            {"tool": "battery", "args": {}},
            {"tool": "storage", "args": {}},
        ],
    ),
]


@dataclass
class WorkflowStep:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    condition: str = ""
    retries: int = 0
    continue_on_error: bool = False


class WorkflowEngine:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.store = ctx.store
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._active: Optional[str] = None
        self._lock = threading.RLock()
        self._ensure_presets()

    def _ensure_presets(self) -> None:
        for name, steps in PRESET_ROUTINES:
            existing = self.store.query_one("SELECT id FROM routines WHERE name=? COLLATE NOCASE", (name,))
            if not existing:
                self.store.execute(
                    "INSERT INTO routines(name, steps, enabled, created) VALUES(?,?,1,?)",
                    (name, json.dumps(steps, ensure_ascii=False), now_iso()),
                )

    def list_routines(self) -> List[Dict[str, Any]]:
        rows = self.store.query("SELECT * FROM routines ORDER BY name")
        for row in rows:
            with contextlib.suppress(ValueError):
                row["steps_parsed"] = json.loads(row["steps"])
            row["enabled"] = bool(row["enabled"])
        return rows

    def save_routine(self, name: str, steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        name = (name or "").strip()[:80]
        if not name:
            return {"ok": False, "error": "Routine needs a name"}
        if not steps:
            return {"ok": False, "error": "Routine has no steps"}
        for step in steps:
            if self.ctx.tools.get(str(step.get("tool", ""))) is None:
                return {"ok": False, "error": f"Unknown tool in routine: {step.get('tool')}"}
        self.store.execute(
            "INSERT INTO routines(name, steps, enabled, created) VALUES(?,?,1,?)",
            (name, json.dumps(list(steps), ensure_ascii=False), now_iso()),
        )
        self.store.audit("automation", "save-routine", name, "low")
        return {"ok": True, "message": f"Routine '{name}' saved"}

    def delete_routine(self, routine_id: int) -> Dict[str, Any]:
        self.store.execute("DELETE FROM routines WHERE id=?", (routine_id,))
        self.store.audit("automation", "delete-routine", str(routine_id), "medium")
        return {"ok": True, "message": "Routine deleted"}

    def toggle_routine(self, routine_id: int, enabled: bool) -> Dict[str, Any]:
        self.store.execute("UPDATE routines SET enabled=? WHERE id=?", (1 if enabled else 0, routine_id))
        return {"ok": True}

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    @property
    def active(self) -> Optional[str]:
        return self._active

    def reset(self) -> None:
        self._cancel.clear()
        self._pause.clear()
        self._active = None

    def _condition_ok(self, condition: str, last_result: Optional[ToolResult]) -> bool:
        condition = (condition or "").strip().lower()
        if not condition:
            return True
        if condition == "last_ok":
            return bool(last_result and last_result.ok)
        if condition == "last_failed":
            return bool(last_result and not last_result.ok)
        if condition == "has_data":
            return bool(last_result and last_result.data)
        return True

    def run_steps(self, name: str, steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._active:
                return {"ok": False, "error": f"Another routine is running: {self._active}"}
            self._active = name
            self.reset()
        started = time.time()
        results: List[Dict[str, Any]] = []
        last_result: Optional[ToolResult] = None
        self.ctx.store.audit("automation", "run-start", name, "low")
        self.ctx.emit("workflow", state="running", name=name, progress=0.0)
        try:
            for index, raw_step in enumerate(steps):
                if self._cancel.is_set():
                    reason = "cancelled"
                    break
                while self._pause.is_set() and not self._cancel.is_set():
                    time.sleep(0.2)
                if self._cancel.is_set():
                    reason = "cancelled"
                    break
                step = WorkflowStep(
                    tool=str(raw_step.get("tool", "")),
                    args=dict(raw_step.get("args", {})),
                    condition=str(raw_step.get("condition", "")),
                    retries=max(0, min(int(raw_step.get("retries", 0) or 0), 3)),
                    continue_on_error=bool(raw_step.get("continue_on_error", False)),
                )
                if not self._condition_ok(step.condition, last_result):
                    results.append({"tool": step.tool, "ok": True, "skipped": True, "message": "condition not met"})
                    continue
                attempts = 0
                result: Optional[ToolResult] = None
                while attempts <= step.retries:
                    result = self.ctx.tools.execute(step.tool, step.args)
                    if result.ok or result.needs_confirmation:
                        break
                    attempts += 1
                if result and result.needs_confirmation:
                    approved = self.ctx.confirm(
                        f"Routine '{name}' step {index + 1} needs confirmation.\n\n{result.message}",
                        result.message,
                    )
                    if approved:
                        result = self.ctx.tools.execute(step.tool, {**step.args, "_confirmed": True})
                    else:
                        results.append({"tool": step.tool, "ok": False, "message": "cancelled by user"})
                        reason = "cancelled"
                        break
                assert result is not None
                results.append(
                    {"tool": step.tool, "ok": result.ok, "message": result.render()[:1200]}
                )
                last_result = result
                self.ctx.emit(
                    "workflow", state="running", name=name,
                    progress=round((index + 1) / max(1, len(steps)), 3),
                    step=step.tool,
                )
                if not result.ok and not step.continue_on_error:
                    reason = "failed"
                    break
            else:
                reason = "completed"
            if self._cancel.is_set():
                reason = "cancelled"
        except Exception as exc:  # noqa: BLE001
            LOG.error("workflow crashed", exc)
            reason = f"error: {exc}"
        finally:
            with self._lock:
                self._active = None
            self.reset()
        lines = [f"Routine '{name}' {reason} in {time.time() - started:.1f}s"]
        for index, item in enumerate(results, start=1):
            status = "✓" if item["ok"] else "✗"
            if item.get("skipped"):
                status = "–"
            lines.append(f"{status} {index}. {item['tool']}: {item['message'][:300]}")
        message = "\n".join(lines)
        self.ctx.store.audit("automation", "run-finish", f"{name}: {reason}", "medium")
        self.ctx.emit("workflow", state="idle", name=name, progress=1.0, reason=reason)
        return {
            "ok": reason == "completed",
            "message": message,
            "reason": reason,
            "results": results,
        }

    def run_by_name(self, name: str) -> Dict[str, Any]:
        needle = normalize_arabic((name or "").lower()).strip()
        routines = self.list_routines()
        target = None
        for routine in routines:
            if not routine["enabled"]:
                continue
            routine_name = normalize_arabic(str(routine["name"]).lower())
            if needle and (needle in routine_name or routine_name in needle):
                target = routine
                break
        if target is None:
            return {"ok": False, "error": f"No routine matched '{name}'"}
        return self.run_steps(target["name"], target.get("steps_parsed", []))

    def save_recording(self, name: str) -> Dict[str, Any]:
        steps = list(self.ctx.recording_steps)
        if not steps:
            return {"ok": False, "error": "No recorded steps. Enable recording first."}
        result = self.save_routine(name, steps)
        if result.get("ok"):
            self.ctx.recording_steps.clear()
            self.ctx.config.set("recording", False)
        return result


# --------------------------------------------------------------------------
# Intelligence orchestrator
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Boshra, a private, offline-first assistant running on the user's Android device.
Rules you must follow:
- Answer in the user's language (Arabic or English). Keep replies concise and useful.
- You have no cloud access unless a tool result is explicitly provided. Never pretend to browse.
- When local context passages are supplied, ground your answer in them and cite [Source: title, page].
- Never invent document contents, page numbers, URLs, device facts, or permissions.
- You cannot execute code. Suggest concrete tool commands instead.
- For medical, safety or legal topics, advise contacting qualified local services.
- Refuse to reveal secrets or credentials, and never ask for passwords.
"""
CANCEL_WORDS = {"stop", "quiet", "silence", "cancel", "halt", "توقف", "اسكت", "اصمت", "ألغ", "الغاء", "إلغاء"}
CONFIRM_WORDS = {"confirm", "yes", "proceed", "do it", "go ahead", "نفذ", "نفّذ", "تأكيد", "اكد", "أكد", "نعم", "تابع"}
DENY_WORDS = {"no", "cancel", "abort", "لا", "الغاء", "إلغاء"}


class IntelligenceOrchestrator:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx
        self.llm = ctx.llm
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self.last_lang = "en"
        self._processing = False
        self._last_cleanup = 0.0

    # -- conversation storage -------------------------------------------
    def _ensure_conversation(self, first_text: str) -> int:
        conversation_id = int(self.ctx.config.last_conversation_id or 0)
        row = self.ctx.store.query_one("SELECT id FROM conversations WHERE id=?", (conversation_id,))
        if conversation_id and row:
            return conversation_id
        title = first_text.strip()[:60] or "Conversation"
        conversation_id = self.ctx.store.execute(
            "INSERT INTO conversations(title, created, updated) VALUES(?,?,?)",
            (title, now_iso(), now_iso()),
        )
        self.ctx.config.set("last_conversation_id", conversation_id)
        return conversation_id

    def _save_message(self, role: str, content: str, lang: str) -> None:
        conversation_id = int(self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return
        sealed, enc = self.ctx.store.seal(content)
        self.ctx.store.execute(
            "INSERT INTO messages(convo_id, role, content, enc, lang, ts) VALUES(?,?,?,?,?,?)",
            (conversation_id, role, sealed, enc, lang, now_iso()),
        )
        self.ctx.store.execute(
            "UPDATE conversations SET updated=? WHERE id=?", (now_iso(), conversation_id)
        )
        if role == "user" and self.ctx.proactive is not None:
            with contextlib.suppress(Exception):
                self.ctx.proactive.note_commitment(content)
        if role == "assistant" and self.ctx.compactor is not None:
            with contextlib.suppress(Exception):
                self.ctx.compactor.maybe_compact(conversation_id)

    def history(self, limit: int = 40) -> List[Dict[str, Any]]:
        conversation_id = int(self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return []
        rows = self.ctx.store.query(
            "SELECT * FROM messages WHERE convo_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        )
        messages = []
        for row in reversed(rows):
            messages.append(
                {
                    "role": row["role"],
                    "content": self.ctx.store.unseal(row["content"], row["enc"]),
                    "lang": row["lang"],
                    "ts": row["ts"],
                }
            )
        return messages

    def new_conversation(self, title: str = "New conversation") -> int:
        conversation_id = self.ctx.store.execute(
            "INSERT INTO conversations(title, created, updated) VALUES(?,?,?)",
            (title[:60], now_iso(), now_iso()),
        )
        self.ctx.take_pending()
        self.ctx.config.set("last_conversation_id", conversation_id)
        return conversation_id

    def drop_trailing_assistants(self) -> int:
        """Remove replies after the latest user turn before regenerating."""
        conversation_id = int(self.ctx.config.last_conversation_id or 0)
        if not conversation_id:
            return 0
        row = self.ctx.store.query_one(
            "SELECT COALESCE(MAX(id), 0) AS last_user FROM messages "
            "WHERE convo_id=? AND role='user'",
            (conversation_id,),
        )
        cutoff = int((row or {}).get("last_user") or 0)
        if not cutoff:
            return 0
        count_row = self.ctx.store.query_one(
            "SELECT COUNT(*) AS count FROM messages "
            "WHERE convo_id=? AND role='assistant' AND id>?",
            (conversation_id, cutoff),
        )
        removed = int((count_row or {}).get("count") or 0)
        if removed:
            self.ctx.store.execute(
                "DELETE FROM messages WHERE convo_id=? AND role='assistant' AND id>?",
                (conversation_id, cutoff),
            )
            self.ctx.store.audit("chat", "regenerate", str(removed), "low")
        return removed

    def cancel(self) -> None:
        self.stop_event.set()
        with contextlib.suppress(Exception):
            self.ctx.tts.stop()
        if self.ctx.voice is not None:
            with contextlib.suppress(Exception):
                self.ctx.voice.stop()
        self.ctx.emit("state", state="idle")
        self.ctx.emit("toast", text="Stopped.")

    # -- text helpers ----------------------------------------------------
    def _lang_for(self, text: str) -> str:
        if self.ctx.config.language in ("ar", "en"):
            return self.ctx.config.language
        return detect_language(text)

    def _is_confirm(self, text: str) -> bool:
        cleaned = normalize_arabic(text.strip().lower().strip(".!؟?"))
        return cleaned in CONFIRM_WORDS

    def _is_deny(self, text: str) -> bool:
        cleaned = normalize_arabic(text.strip().lower().strip(".!؟?"))
        return cleaned in DENY_WORDS

    def _context_block(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        memories = self.ctx.memory.search(query, k=3)
        for memory in memories:
            parts.append(
                f"[MEMORY · {memory['category']} · {memory['source']} · {human_ts(memory['updated'])}]\n"
                f"{memory['content'][:900]}"
            )
        documents = self.ctx.vault.documents()
        if documents:
            passages = self.ctx.vault.search(query, k=int(self.ctx.resources.profile["retrieval_k"]))
            for passage in passages:
                if passage["score"] < 0.1:
                    continue
                page = f", page {passage['page']}" if passage["page"] else ""
                parts.append(
                    f"[VAULT · {passage['title']}{page}]\n{passage['text'][:1200]}"
                )
                sources.append({"title": passage["title"], "page": passage["page"], "doc_id": passage["doc_id"]})
        return "\n\n".join(parts), sources[:6]

    def _history_block(self) -> str:
        if self.ctx.compactor is not None:
            with contextlib.suppress(Exception):
                block = self.ctx.compactor.history_block()
                if block:
                    return block
        messages = self.history(limit=10)
        lines = []
        for message in messages[:-1][-8:]:
            content = message["content"].replace("\n", " ")[:500]
            lines.append(f"{message['role'].upper()}: {content}")
        return "\n".join(lines)

    def _remember_conversation_digest(self, conversation_id: int) -> None:
        try:
            rows = self.ctx.store.query(
                "SELECT role, content, enc FROM messages WHERE convo_id=? ORDER BY id DESC LIMIT 30",
                (conversation_id,),
            )
            if len(rows) < 30:
                return
            lines = []
            for row in reversed(rows[-12:]):
                content = self.ctx.store.unseal(row["content"], row["enc"]).replace("\n", " ")
                lines.append(f"{row['role']}: {content[:240]}")
            self.ctx.memory.add(
                "conversation",
                "Conversation digest:\n" + "\n".join(lines),
                key=f"digest:{conversation_id}",
                source="conversation",
                tags="auto-digest",
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"conversation digest failed: {exc}")

    # -- routing ---------------------------------------------------------
    def _route(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        raw = text.strip()
        lowered = raw.lower()
        norm = normalize_arabic(lowered)

        def has(*words: str) -> bool:
            return any(normalize_arabic(word) in norm for word in words)

        if norm.strip(" !.؟?") in (
            "daily briefing", "my day", "today's agenda", "plan my day",
            normalize_arabic("موجز اليوم"), normalize_arabic("خطط يومي"),
        ):
            return ("tool", {"tool": "daily_briefing", "args": {"lang": self._lang_for(raw)}})

        # navigation and status
        if re.fullmatch(r"(help|مساعدة|ماذا تستطيع.*|capabilities?)", norm.strip(" !.؟?")):
            self.ctx.emit("navigate", screen="home")
            return ("chat", {})
        if re.search(r"(show|open|اعرض|افتح)\s+(skills|المهارات)", norm) or norm.strip(" ؟?") in ("skills", "المهارات"):
            self.ctx.emit("navigate", screen="skills")
            return ("tool", {"tool": "skill_list", "args": {}})
        if re.search(r"(privacy|الخصوصية)", norm) and len(norm) < 40:
            self.ctx.emit("navigate", screen="privacy")
            return ("nav", {"screen": "privacy", "message": "Privacy Center opened."})
        if re.search(r"(diagnostic|تشخيص|صحة النظام)", norm):
            self.ctx.emit("navigate", screen="diagnostics")
            return ("tool", {"tool": "diagnostics", "args": {}})
        if re.search(r"(memory|الذاكرة)", norm) and re.match(r"^(show|open|اعرض|افتح|الذاكرة)", norm):
            self.ctx.emit("navigate", screen="memory")
            return ("tool", {"tool": "memory_search", "args": {"query": raw}})
        if re.fullmatch(r"(?:(?:show|open|اعرض|افتح)\s+)?(?:my\s+|the\s+)?(?:vault|knowledge|documents|الخزنه|المستندات)", norm.strip(" !.؟?")):
            self.ctx.emit("navigate", screen="vault")
            return ("tool", {"tool": "vault_list", "args": {}})
        if re.search(r"\b(settings|الإعدادات|الاعدادات)\b", norm) and len(norm) < 30:
            self.ctx.emit("navigate", screen="settings")
            return ("nav", {"screen": "settings", "message": "Settings opened."})
        if has("emergency", "طوارئ", "إسعاف", "اسعاف", "حريق", "زلزال", "فيضان", "اختناق", "نزيف", "cpr"):
            self.ctx.emit("navigate", screen="emergency")
            topic = raw
            return ("tool", {"tool": "emergency", "args": {"query": topic}})
        if (re.search(r"^(?:(?:what(?:'s| is)?|tell me|show|current|the)\s+)*(?:time|date|clock)\b", norm)
                or re.fullmatch(r"(?:what (?:time|date) is it|ما الوقت|كم الساعه|ما التاريخ|الوقت|الساعه|التاريخ)[ ؟?!.]*", norm)) and len(norm) < 60:
            return ("tool", {"tool": "time_date", "args": {"query": raw}})
        if re.search(r"(battery|بطارية|شاحن)", norm):
            return ("tool", {"tool": "battery", "args": {}})
        if re.search(r"(storage|تخزين|مساحة)", norm):
            return ("tool", {"tool": "storage", "args": {}})
        if re.search(r"(device info|معلومات الجهاز|مواصفات الجهاز)", norm):
            return ("tool", {"tool": "device_info", "args": {}})
        match = re.search(r"^(?:open|launch|افتح|شغل|شغّل)\s+(?:the\s+|تطبيق\s+)?(.+?)(?:\s+app)?[.!]?$", norm)
        if match and len(match.group(1)) < 40:
            return ("tool", {"tool": "open_app", "args": {"app": match.group(1).strip()}})
        match = re.search(r"(?:theme|مظهر|ثيم)\s*[:=]?\s*(aurora|matrix|neon|lava|darkblue|minimal|glass)", norm)
        if match:
            return ("tool", {"tool": "theme_set", "args": {"theme": match.group(1)}})
        match = re.search(r"\b(lite|balanced|performance)\b", norm)
        if match and re.search(r"(profile|وضع|أداء|موارد)", norm):
            return ("tool", {"tool": "profile_set", "args": {"profile": match.group(1)}})

        # memory
        match = re.search(r"^(?:remember|please remember|تذكر|تذكري|احفظ|احفظي)\s+(?:that\s+|أن\s+|ان\s+)?(.+)$", norm, re.DOTALL)
        if match:
            return ("remember", {"content": re.sub(r"^(?:please\s+)?remember\s+(?:that\s+)?", "", raw, count=1, flags=re.IGNORECASE)
                    if re.match(r"^(?:please\s+)?remember\b", raw, re.IGNORECASE)
                    else raw.split(maxsplit=1)[1]})
        match = re.search(r"(?:what do you remember|do you remember|ماذا تتذكر|ماذا تذكر|هل تتذكر)\s*(?:about|عن|حول)?\s*(.*)", norm)
        if match:
            return ("tool", {"tool": "memory_search", "args": {"query": match.group(1).strip() or raw}})

        # tasks & notes & reminders
        match = re.search(r"^(?:add|create|أضف|اضف|أضيفي)\s+(?:a\s+)?(?:task|مهم[ةه])\s*[:\-]?\s*(.+)$", raw, re.IGNORECASE)
        if match:
            return ("tool", {"tool": "task_add", "args": {"title": match.group(1)}})
        if re.search(r"^(?:list|show|اعرض|لائحة)\s+(?:my\s+)?(?:tasks|المهام|مهامي)$", norm) or norm in ("tasks", "المهام"):
            return ("tool", {"tool": "task_list", "args": {}})
        match = re.search(r"^(?:done|complete|أنجز|اكمل|أكمل)\s+(?:task\s+)?#?(\d+)$", norm)
        if match:
            return ("tool", {"tool": "task_done", "args": {"id": int(match.group(1))}})
        match = re.search(r"^(?:note|ملاحظ[ةه]|اكتب ملاحظ[ةه]|اكتبي ملاحظ[ةه])\s*[:\-]?\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
        if match:
            body = match.group(1)
            return ("tool", {"tool": "note_add", "args": {"title": body[:50], "body": body}})
        match = re.search(r"remind me (?:to )?(.+?) (?:in|after) (\d+)\s*(minutes?|min|hours?)\b", raw, re.IGNORECASE)
        if not match:
            match_ar = re.search(r"(?:ذكرني|ذكّرني)\s+(?:ب|بـ)?\s*(.+?)\s+(?:بعد|خلال)\s*(\d+)\s*(دقيقة|دقائق|ساعة|ساعات)", norm)
            if match_ar:
                minutes = int(match_ar.group(2)) * (60 if "ساع" in match_ar.group(3) else 1)
                return ("tool", {"tool": "reminder_add", "args": {"title": match_ar.group(1), "minutes": minutes}})
        elif match:
            minutes = int(match.group(2)) * (60 if match.group(3).startswith("hour") else 1)
            return ("tool", {"tool": "reminder_add", "args": {"title": match.group(1), "minutes": minutes}})
        if re.search(r"^(?:list|show|اعرض)\s+(?:my\s+)?(?:notes|الملاحظات)$", norm):
            return ("tool", {"tool": "note_list", "args": {}})
        if re.search(r"^(?:list|show|اعرض)\s+(?:my\s+)?(?:reminders|التذكيرات)$", norm):
            return ("tool", {"tool": "reminder_list", "args": {}})
        match = re.search(r"(?:focus|ركّز|ركز)\s*(?:for\s*)?(\d+)\s*(?:min|minutes|دقيقة|دقائق)?", norm)
        if match:
            return ("tool", {"tool": "focus_start", "args": {"minutes": int(match.group(1))}})

        # documents
        if re.search(r"(summari[sz]e|لخص|تلخيص|ملخص)", norm) and re.search(r"(latest|last|آخر|أحدث|اخر).*(pdf|document|ملف|مستند)", norm):
            return ("latest_summary", {})
        if re.search(r"(read|اقرأ|اعرض|افتح).*(latest|last|آخر|أحدث).*(pdf|word|docx|document|ملف|مستند)", norm):
            return ("latest_read", {})
        match = re.search(r"(?:summari[sz]e|lakees|لخص|تلخيص)\s+(?:the\s+|document\s+|ملف\s+|مستند\s+)?(.+)$", raw, re.IGNORECASE)
        if match:
            return ("tool", {"tool": "vault_summarize", "args": {"document": match.group(1).strip()}})
        match = re.search(r"(?:import|استورد|أضف|اضف)\s+(?:document|file|ملف|مستند)\s*[:\-]?\s*(.+)$", raw, re.IGNORECASE)
        if match:
            return ("tool", {"tool": "vault_import", "args": {"path": match.group(1).strip()}})
        match = re.search(r"(?:search|find|ابحث|جد|ابحثي)\s+(?:my\s+|in\s+)?(?:files|ملفاتي|الملفات)\s+(?:for\s+|عن\s+)?(.+)$", norm)
        if match:
            return ("tool", {"tool": "file_search", "args": {"query": match.group(1).strip()}})
        match = re.search(r"(?:search|ابحث)\s+(?:in\s+|في\s+)?(?:the\s+|my\s+)?(?:vault|knowledge|documents|الخزنة|المستندات)\s+(?:for\s+|عن\s+)?(.+)$", norm)
        if match:
            return ("tool", {"tool": "vault_search", "args": {"query": match.group(1).strip()}})
        match = re.search(r"(?:ask (?:my )?(?:documents|vault)|answer from (?:my )?(?:documents|vault)|اسأل (?:مستنداتي|الخزنة))\s*[:\-]?\s*(.+)$", norm)
        if match:
            return ("tool", {"tool": "vault_answer", "args": {"query": match.group(1).strip()}})
        if re.search(r"(organi[sz]e|tidy|رتب|نظّم|نظم).*(files|downloads|ملفات|تحميلات)", norm):
            return ("tool", {"tool": "organize_files", "args": {"folder": ""}})

        # online
        match = re.search(r"(?:translate|ترجم)\s+(?:this\s+|هذا\s+|النص\s+)?[:\-]?\s*(.+?)(?:\s+(?:to|إلى|الى)\s+(arabic|english|العربية|الإنجليزية|الانجليزية))?$", raw, re.IGNORECASE | re.DOTALL)
        if match:
            target = (match.group(2) or "").lower()
            target_lang = "ar" if target in ("arabic", "العربية") else ("en" if target in ("english", "الإنجليزية", "الانجليزية") else "")
            return ("tool", {"tool": "translate", "args": {"text": match.group(1).strip(), "target": target_lang}})
        match = re.search(r"(?:weather|طقس|الجو)\s*(?:in|for|في|ب)?\s*(.+)$", norm)
        if match:
            return ("tool", {"tool": "weather", "args": {"city": match.group(1).strip()}})
        if re.search(r"\b(news|headlines|أخبار|اخبار)\b", norm):
            return ("tool", {"tool": "news", "args": {}})
        match = re.search(r"(?:search the web(?: for)?|web search(?: for)?|google|ابحث في (?:الإنترنت|الانترنت|الويب) عن)\s+(.+)$", norm)
        if match:
            return ("tool", {"tool": "web_search", "args": {"query": match.group(1).strip()}})

        # developer tools
        match = re.search(r"(?:explain code|اشرح الكود)[:\s]*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return ("tool", {"tool": "code_explain", "args": {"code": match.group(1).strip()}})
        match = re.search(r"(?:review code|review security|راجع الكود|راجع الأمان)[:\s]*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return ("tool", {"tool": "secure_review", "args": {"code": match.group(1).strip()}})

        # intelligence surfaces: plans, recall, backups, briefs, compaction
        match = re.search(r"^(?:plan|خطة|خطط)\s*[:\-]\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
        if match:
            return ("tool", {"tool": "plan_create", "args": {"goal": match.group(1).strip()}})
        if re.search(r"(?:run|execute|نفّذ|نفذ)\s+(?:the\s+|ال)?(?:plan|خطة)", norm):
            number = re.search(r"#?(\d+)", norm)
            return ("tool", {"tool": "plan_run",
                             "args": {"plan_id": int(number.group(1)) if number else 0}})
        if norm.strip(" !.؟?") in ("plans", "my plans", "الخطط", "خططي", "قائمة الخطط"):
            return ("tool", {"tool": "plan_list", "args": {}})
        match = re.search(r"^(?:recall|استرجع|استرجاع|استدعِ)\s*[:\-]?\s*(.+)$", raw, re.IGNORECASE)
        if match:
            return ("tool", {"tool": "recall", "args": {"query": match.group(1).strip()}})
        if re.search(r"^(?:reindex|أعد فهرسة|افهرس|فهرس)\b", norm):
            return ("tool", {"tool": "recall_reindex", "args": {}})
        if re.search(r"^(?:list|show|اعرض)\s+(?:my\s+)?(?:backups|النسخ الاحتياطية|النسخ)$", norm):
            return ("tool", {"tool": "backup_list", "args": {}})
        if re.match(r"^(?:backup|create backup|نسخة احتياطية|أنشئ نسخة|انشئ نسخة)", norm):
            return ("tool", {"tool": "backup_create", "args": {}})
        if re.match(r"^(?:restore|استعد|استعادة)\b", norm):
            return ("tool", {"tool": "backup_restore", "args": {}})
        if re.match(r"^(?:digest|brief me|موجز|ملخص اليوم|مراجعة المساء|مراجعة اليوم)", norm):
            kind = "evening" if re.search(r"(?:evening|مساء|مراجعة)", norm) else "morning"
            return ("tool", {"tool": "digest", "args": {"kind": kind}})
        if re.search(r"^(?:follow[ -]?ups?|متابعات|متابعاتي)\b", norm):
            return ("tool", {"tool": "commitments", "args": {}})
        if re.search(r"^(?:compact|compress|اضغط المحادثة|لخّص المحادثة|لخص المحادثة)", norm):
            return ("tool", {"tool": "compact", "args": {}})
        if re.search(r"(?:floating (?:assistant|orb|bubble)|المساعد العائم|الفقاعة العائمة)", norm):
            action = "stop" if re.search(r"(?:off|stop|أوقف|ايقاف|إيقاف|اخف|أخف)", norm) else "start"
            return ("tool", {"tool": "float", "args": {"action": action}})

        # skills & automation
        match = re.search(r"(?:run skill|شغل المهارة|نفذ المهارة)\s+([a-z0-9_.]+)(?:\s+(.+))?", norm)
        if match:
            return ("tool", {"tool": "skill_execute", "args": {"skill": match.group(1), "command": (match.group(2) or "").strip()}})
        if re.search(r"(start|begin|ابدأ|شغل).*(programming|session|جلسة|برمجة)", norm):
            return ("routine", {"name": "programming session"})
        if re.search(r"(morning briefing|موجز الصباح|briefing صباح)", norm):
            return ("routine", {"name": "morning briefing"})
        if re.search(r"(device check|فحص الجهاز)", norm):
            return ("routine", {"name": "device check"})
        match = re.search(r"(?:run routine|شغل الروتين)\s+(.+)$", norm)
        if match:
            return ("routine", {"name": match.group(1).strip()})
        return None

    # -- processing ------------------------------------------------------
    _FALLBACK_MARKERS = (
        "open-ended generation is unavailable",
        "لا أستطيع توليد نص حر عام",
    )

    def _online_fallback(self, query: str, final_text: str, lang: str) -> Optional[str]:
        if self.ctx.config.online_mode == "local":
            return None
        if not any(marker in final_text for marker in self._FALLBACK_MARKERS):
            return None
        result = self.ctx.tools.execute("web_search", {"query": query[:240]})
        if not result.ok:
            note = (
                "تعذر البحث على الإنترنت: " if lang == "ar" else "Internet search failed: "
            )
            return final_text + "\n\n" + note + str(result.error or "")
        header = (
            "لم أجد إجابة محلية، لذلك بحثت على الإنترنت. النتائج الخارجية غير مُتحقَّق منها:\n\n"
            if lang == "ar" else
            "No local answer was available, so I searched the web. External results are unverified:\n\n"
        )
        return header + result.render()

    def process(
        self,
        text: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        record_user: bool = True,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        with self._lock:
            if self._processing:
                return "Boshra is still answering. Say stop to interrupt."
            self._processing = True
        self.stop_event.clear()
        state_cb = on_state or (lambda state: self.ctx.emit("state", state=state))
        token_cb = on_token or (lambda token: None)
        final_text = ""
        meta: Dict[str, Any] = {}
        online_used = False
        try:
            lang = self._lang_for(text)
            self.last_lang = lang
            conversation_id = self._ensure_conversation(text)
            if record_user:
                self._save_message("user", text, lang)
            state_cb("thinking")
            self.ctx.emit("state", state="thinking")
            self.ctx.store.audit("chat", "user-message", text[:300], "low")

            # global stop command
            if normalize_arabic(text.lower().strip(" !.؟?")) in CANCEL_WORDS:
                self.cancel()
                final_text = "تم الإيقاف." if lang == "ar" else "Stopped."
                self._save_message("assistant", final_text, lang)
                meta = {"intent": "stop"}
                return final_text

            # pending destructive confirmation
            pending = self.ctx.pending
            if pending:
                if self._is_confirm(text):
                    taken = self.ctx.take_pending()
                    if taken:
                        result = self.ctx.tools.execute(
                            taken["tool"], {**taken["args"], "_confirmed": True}
                        )
                        final_text = result.render()
                        meta = {"intent": "confirmed-tool", "tool": taken["tool"]}
                        self.ctx.tts.speak(final_text, lang)
                        self._save_message("assistant", final_text, lang)
                        return final_text
                elif self._is_deny(text):
                    self.ctx.take_pending()
                    final_text = "تم الإلغاء." if lang == "ar" else "Cancelled."
                    self._save_message("assistant", final_text, lang)
                    return final_text

            route = self._route(text)
            if route is not None:
                kind, payload = route
                if kind == "nav":
                    final_text = str(payload.get("message", "Opened."))
                    self.ctx.tts.speak(final_text, lang)
                    self._save_message("assistant", final_text, lang)
                    meta = {"intent": "navigation"}
                    return final_text
                if kind == "routine":
                    result = self.ctx.automation.run_by_name(str(payload.get("name", ""))) if self.ctx.automation else {"ok": False, "error": "Automation unavailable"}
                    final_text = str(result.get("message") or result.get("error") or "")
                    meta = {"intent": "routine", "result": result}
                    self.ctx.tts.speak(final_text, lang)
                    self._save_message("assistant", final_text, lang)
                    return final_text
                if kind == "remember":
                    content = str(payload.get("content", "")).strip()
                    if not content:
                        final_text = "ماذا تريد أن أتذكر؟" if lang == "ar" else "What should I remember?"
                    else:
                        result = self.ctx.memory.add("semantic", content, source="user")
                        if result.get("ok"):
                            final_text = (
                                f"حفظت ذلك محلياً: {content[:200]}\n(المصدر: أنت)"
                                if lang == "ar" else
                                f"Remembered locally: {content[:200]}\n(Source: you)"
                            )
                        else:
                            final_text = str(result.get("error", "Memory refused"))
                    self.ctx.tts.speak(final_text, lang)
                    self._save_message("assistant", final_text, lang)
                    meta = {"intent": "remember"}
                    return final_text
                if kind == "latest_summary":
                    result = self._latest_summary(lang)
                    final_text = result.render()
                    meta = {"intent": "latest-summary"}
                    self.ctx.tts.speak(final_text, lang)
                    self._save_message("assistant", final_text, lang)
                    return final_text
                if kind == "latest_read":
                    result = self._latest_read(lang)
                    final_text = result.render()
                    meta = {"intent": "latest-read"}
                    self.ctx.tts.speak(final_text[:1200], lang)
                    self._save_message("assistant", final_text, lang)
                    return final_text
                if kind == "tool":
                    tool = str(payload.get("tool", ""))
                    args = dict(payload.get("args", {}))
                    result = self.ctx.tools.execute(tool, args)
                    final_text = result.render()
                    if result.needs_confirmation:
                        hint = (
                            "\n\nهل تريد التنفيذ؟ أجب بـ «تأكيد» أو «إلغاء»."
                            if lang == "ar" else
                            "\n\nDo you want to proceed? Reply 'confirm' or 'cancel'."
                        )
                        final_text += hint
                    if result.sources:
                        meta["sources"] = result.sources
                    meta.update({"intent": "tool", "tool": tool, "ok": result.ok})
                    self.ctx.tts.speak(final_text, lang)
                    self._save_message("assistant", final_text, lang)
                    return final_text

            # conversational path with local RAG
            context_block, sources = self._context_block(text)
            history_block = self._history_block()
            prompt_parts = []
            if context_block:
                prompt_parts.append("LOCAL CONTEXT (retrieved on-device):\n" + context_block)
            if history_block:
                prompt_parts.append("RECENT CONVERSATION:\n" + history_block)
            prompt_parts.append("USER REQUEST:\n" + text)
            prompt = "\n\n".join(prompt_parts)
            time_info = tool_time(self.ctx, {"query": text}).message
            max_tokens = int(self.ctx.resources.profile["max_tokens"])
            collected: List[str] = []
            engine_used = self.llm.engine_name
            state_cb("thinking")
            for token in self.llm.generate_stream(
                prompt, SYSTEM_PROMPT, max_tokens, 0.65, self.stop_event,
                context=context_block, lang=lang, time_info=time_info,
            ):
                if self.stop_event.is_set():
                    break
                if not collected:
                    state_cb("speaking")
                    self.ctx.emit("state", state="speaking")
                collected.append(token)
                token_cb(token)
                if self.ctx.tts:
                    self.ctx.tts.feed(token, lang)
            final_text = "".join(collected).strip()
            if self.ctx.tts:
                self.ctx.tts.flush()
            if not final_text:
                final_text = "تعذر توليد رد." if lang == "ar" else "I could not generate a response."
            if not self.stop_event.is_set():
                online_text = self._online_fallback(text, final_text, lang)
                if online_text is not None:
                    final_text = online_text
                    online_used = True
                    if self.ctx.tts:
                        with contextlib.suppress(Exception):
                            self.ctx.tts.stop()
                        self.ctx.tts.speak(final_text[:700], lang)
            if sources and "Sources:" not in final_text:
                formatted = ", ".join(
                    f"{s['title']}" + (f" p.{s['page']}" if s.get("page") else "") for s in sources
                )
                final_text += ("\n\nالمصادر المحلية: " if lang == "ar" else "\n\nLocal sources: ") + formatted
                token_cb(("\n\nالمصادر المحلية: " if lang == "ar" else "\n\nLocal sources: ") + formatted)
            meta = {"intent": "chat-online" if online_used else "chat", "engine": engine_used, "sources": sources}
            self._save_message("assistant", final_text, lang)
            self._remember_conversation_digest(conversation_id)
            self._periodic_cleanup()
            self.llm.ensure_policy()
            return final_text
        except Exception as exc:  # noqa: BLE001
            LOG.error("orchestrator failed", exc)
            final_text = (
                "حدث خطأ محلي أثناء المعالجة. التفاصيل في ملف الأخطاء."
                if self.last_lang == "ar" else
                "A local error interrupted processing. Details were appended to the error log."
            )
            _append_error_dump("orchestrator", traceback.format_exc())
            meta = {"intent": "error", "error": str(exc)}
            return final_text
        finally:
            with self._lock:
                self._processing = False
            state_cb("idle")
            self.ctx.emit("state", state="idle")
            if on_done:
                try:
                    on_done(final_text, meta)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(f"on_done callback failed: {exc}")

    def _periodic_cleanup(self) -> None:
        if time.time() - self._last_cleanup < 3600:
            return
        self._last_cleanup = time.time()
        retention = self.ctx.config.retention_days
        if retention > 0:
            self.ctx.memory.purge_expired(retention)

    def _latest_summary(self, lang: str) -> ToolResult:
        search = self.ctx.tools.execute("file_search", {"extension": "pdf", "limit": 5})
        files = (search.data or {}).get("files", [])
        if not files:
            return ToolResult(False, error="لا يوجد ملف PDF في المجلدات المعتمدة." if lang == "ar" else "No PDF found in approved folders.")
        imported = self.ctx.tools.execute("vault_import", {"path": files[0]["path"]})
        if not imported.ok:
            return imported
        doc_id = (imported.data or {}).get("doc_id")
        return self.ctx.tools.execute("vault_summarize", {"document": str(doc_id)})

    def _latest_read(self, lang: str) -> ToolResult:
        search = self.ctx.tools.execute("file_search", {"extension": "pdf", "limit": 5})
        files = (search.data or {}).get("files", [])
        if not files:
            return ToolResult(False, error="لا يوجد ملف PDF." if lang == "ar" else "No PDF found.")
        return self.ctx.tools.execute("read_document", {"path": files[0]["path"]})


# ==========================================================================
# SECTION 9 - KIVY INTERFACE, GLSL CORE AND APPLICATION BOOTSTRAP
# ==========================================================================
from kivy.app import App  # noqa: E402 - configure Kivy before importing UI modules
from kivy.clock import Clock  # noqa: E402 - configure Kivy before importing UI modules
from kivy.core.text import LabelBase  # noqa: E402 - configure Kivy before importing UI modules
from kivy.core.window import Window  # noqa: E402 - configure Kivy before importing UI modules
from kivy.graphics import Color, Ellipse, Line, Rectangle, RenderContext, RoundedRectangle  # noqa: E402 - configure Kivy before importing UI modules
from kivy.metrics import dp, sp  # noqa: E402 - configure Kivy before importing UI modules
from kivy.properties import BooleanProperty  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.boxlayout import BoxLayout  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.button import Button  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.floatlayout import FloatLayout  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.gridlayout import GridLayout  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.label import Label  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.modalview import ModalView  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.scrollview import ScrollView  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.slider import Slider  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.spinner import Spinner  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.textinput import TextInput  # noqa: E402 - configure Kivy before importing UI modules
from kivy.uix.widget import Widget  # noqa: E402 - configure Kivy before importing UI modules
from kivy.utils import get_color_from_hex  # noqa: E402 - configure Kivy before importing UI modules

FONT = "Boshra"
HEADER_H = dp(58)
NAV_H = dp(78)
TELEM_W = dp(112)


def _rgba(value: str, alpha: float = 1.0) -> List[float]:
    try:
        red, green, blue, _ = get_color_from_hex(value)
        return [red, green, blue, float(alpha)]
    except Exception:
        return [1.0, 1.0, 1.0, float(alpha)]


PALETTES: Dict[str, Dict[str, str]] = {
    "aurora": {
        "bg": "#04080f", "panel": "#081521", "text": "#e8f7ff", "dim": "#7d9db5",
        "accent": "#38e8ff", "accent2": "#7a5cff", "danger": "#ff4d6d", "good": "#5dffb0",
    },
    "matrix": {
        "bg": "#020803", "panel": "#06140a", "text": "#dcffe6", "dim": "#67a37b",
        "accent": "#42ff87", "accent2": "#0affc3", "danger": "#ff5d5d", "good": "#8affb0",
    },
    "neon": {
        "bg": "#0a0414", "panel": "#160a22", "text": "#f6eaff", "dim": "#a184bb",
        "accent": "#ff3df2", "accent2": "#38c8ff", "danger": "#ff4d6d", "good": "#62ffa8",
    },
    "lava": {
        "bg": "#0d0503", "panel": "#1a0b07", "text": "#fff0e6", "dim": "#b58a76",
        "accent": "#ff7a1a", "accent2": "#ff2d55", "danger": "#ff2d55", "good": "#ffd166",
    },
    "darkblue": {
        "bg": "#030a16", "panel": "#081524", "text": "#e4eefc", "dim": "#7189a6",
        "accent": "#4d8dff", "accent2": "#2de2e6", "danger": "#ff5c7a", "good": "#5cffb1",
    },
    "minimal": {
        "bg": "#0b0c0f", "panel": "#15171c", "text": "#eef0f4", "dim": "#8b909c",
        "accent": "#d8d8d8", "accent2": "#8fa1b8", "danger": "#ff6b6b", "good": "#7bd88f",
    },
    "glass": {
        "bg": "#070d12", "panel": "#0d1a22", "text": "#eaf6ff", "dim": "#88a4b5",
        "accent": "#a7e8ff", "accent2": "#8fd4ff", "danger": "#ff5c7a", "good": "#72ffc5",
    },
}


def _font_pairs() -> Iterator[Tuple[Path, Optional[Path]]]:
    """Yield usable (regular, bold-or-None) font pairs in preference order.

    Bundled assets win, then Android system fonts, then the font shipped with
    Kivy. Every candidate is checked before use, so a missing ``.ttf`` can
    never reach ``LabelBase.register``.
    """
    pairs: List[Tuple[Path, Path]] = []
    try:
        font_folders = _asset_folders("fonts", asset_roots())
    except Exception:  # noqa: BLE001 - asset discovery must not block startup
        font_folders = []
    if FONTS_DIR not in font_folders:
        font_folders.insert(0, FONTS_DIR)
    for folder in font_folders:
        pairs.append((folder / "DejaVuSans.ttf", folder / "DejaVuSans-Bold.ttf"))
        pairs.append((folder / "Boshra.ttf", folder / "Boshra-Bold.ttf"))
        for font in _scan_files(folder, "*.ttf")[:4]:
            pairs.append((font, font.with_name(f"{font.stem}-Bold.ttf")))
    pairs.extend([
        (Path("/system/fonts/NotoSansArabic-Regular.ttf"),
         Path("/system/fonts/NotoSansArabic-Bold.ttf")),
        (Path("/system/fonts/DroidSansArabic.ttf"),
         Path("/system/fonts/DroidSansArabic-Bold.ttf")),
        (Path("/system/fonts/Roboto-Regular.ttf"), Path("/system/fonts/Roboto-Bold.ttf")),
        (Path("/system/fonts/DroidSans.ttf"), Path("/system/fonts/DroidSans-Bold.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ])
    try:
        import kivy

        kivy_fonts = Path(kivy.__file__).resolve().parent / "data" / "fonts"
        pairs.append((kivy_fonts / "Roboto-Regular.ttf", kivy_fonts / "Roboto-Bold.ttf"))
    except Exception:  # noqa: BLE001 - Kivy is present in practice
        pass
    seen: set = set()
    for regular, bold in pairs:
        if not _is_file(regular):
            continue
        key = str(regular)
        if key in seen:
            continue
        seen.add(key)
        yield regular, bold if _is_file(bold) else None


def register_fonts() -> None:
    """Register the UI font, always leaving ``FONT`` bound to a usable family.

    The previous version returned early when the bundled font was absent, but
    widgets still requested the custom family, producing
    ``Label: File 'Boshra.ttf' not found``. Now the family is only advertised
    after a real file was registered; otherwise ``FONT`` switches to Kivy's
    built-in Roboto.
    """
    global FONT
    family = "Boshra"
    pair = next(_font_pairs(), None)
    if pair is None:
        FONT = "Roboto"
        LOG.warning("no usable font file found; using Kivy's built-in Roboto")
        return
    regular, bold = pair
    try:
        LabelBase.register(
            name=family,
            fn_regular=str(regular),
            fn_bold=str(bold or regular),
        )
        FONT = family
    except Exception as exc:  # noqa: BLE001
        FONT = "Roboto"
        LOG.error("custom font registration failed; using Kivy Roboto", exc)
        return
    with contextlib.suppress(Exception):
        # Keep Kivy's default family consistent with the selected font.
        LabelBase.register(
            name="Roboto",
            fn_regular=str(regular),
            fn_bold=str(bold or regular),
        )
    LOG.info(f"UI font: {regular}")


def T(en: str, ar: str) -> str:
    """Logical UI string; rendering helpers apply Arabic shaping once."""
    return ar if _UI_LANG == "ar" else (en if _UI_LANG == "en" else f"{en} · {ar}")


def T1(en: str, ar: str) -> str:
    """Compact label for small controls that cannot fit a bilingual line."""
    return ar if _UI_LANG == "ar" else en


AURORA_FS = """
$HEADER$
uniform vec2 resolution;
uniform float time;
uniform vec2 tilt;
uniform float sphere_state;
uniform float sphere_alpha;
uniform float energy;
uniform vec2 shock_origin;
uniform float shock_time;
uniform float intensity;
uniform float sphere_style;
uniform vec3 sphere_tint;
uniform vec3 sphere_glow;
uniform vec2 sphere_center;
uniform float sphere_radius;
uniform vec3 color_a;
uniform vec3 color_b;
uniform vec3 bg_color;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 4; i++) {
        value += amplitude * vnoise(p);
        p = p * 2.03 + vec2(11.7, 5.3);
        amplitude *= 0.5;
    }
    return value;
}

void main(void) {
    vec2 frag = vec2(frag_modelview_mat * gl_FragCoord).xy;
    vec2 uv = frag / max(resolution, vec2(1.0));
    float aspect = resolution.x / max(resolution.y, 1.0);
    vec2 p = vec2((uv.x - 0.5) * aspect, uv.y - 0.5);
    float t = time;

    vec3 col = mix(bg_color * 0.55, bg_color * 1.25, smoothstep(0.0, 1.0, uv.y));
    col += color_b * 0.06 * (1.0 - uv.y);

    vec2 grid = floor(frag * 0.34);
    float star = step(0.9975, hash21(grid));
    float twinkle = 0.55 + 0.45 * sin(t * 2.1 + hash21(grid + 5.0) * 30.0);
    col += vec3(0.75, 0.92, 1.0) * star * twinkle * 0.55 * smoothstep(0.32, 0.95, uv.y);

    float n1 = fbm(vec2(uv.x * 3.1 + t * 0.045 + tilt.x * 0.55, uv.y * 3.4 - t * 0.052 + tilt.y * 0.45));
    float n2 = fbm(vec2(uv.x * 5.7 - t * 0.032 + tilt.x * 0.3, uv.y * 5.4 + t * 0.028));
    float ribbons = smoothstep(0.30, 0.92, n1 * 0.74 + n2 * 0.44);
    float band = smoothstep(0.02, 0.38, uv.y) * smoothstep(0.92, 0.42, uv.y);
    col += color_a * ribbons * band * 0.55 * intensity;
    col += color_b * pow(1.0 - uv.y, 3.0) * 0.25 * intensity;

    float scan = 0.975 + 0.025 * sin(frag.y * 1.7 + t * 7.0);
    col *= scan;

    vec2 sc = vec2((sphere_center.x - 0.5) * aspect, sphere_center.y - 0.5);
    float radius = max(sphere_radius, 0.01);
    vec2 q = p - sc;
    float dist = length(q);
    float dnorm = dist / radius;
    float mask = smoothstep(1.0, 0.968, dnorm);

    vec3 idle_col = vec3(0.10, 0.86, 0.95);
    vec3 think_col = vec3(1.0, 0.72, 0.18);
    vec3 speak_col = vec3(1.0, 0.16, 0.72);
    vec3 listen_col = vec3(0.36, 1.0, 0.60);
    vec3 state_col = idle_col;
    if (sphere_state > 2.5) {
        state_col = listen_col;
    } else if (sphere_state > 1.5) {
        state_col = speak_col;
    } else if (sphere_state > 0.5) {
        state_col = think_col;
    }
    state_col = mix(state_col, sphere_tint, 0.62);
    if (length(sphere_tint) < 0.01) {
        state_col = mix(state_col, color_a, 0.22);
    }

    float n3 = fbm(q * 7.5 + vec2(t * 0.22, -t * 0.18) + tilt * 0.35);
    float n4 = fbm(q * 13.0 - vec2(t * 0.31, t * 0.27));
    float swirl = 0.5 + 0.5 * sin(n3 * 5.0 + n4 * 3.0 + t * (1.3 + sphere_state * 1.5) + dnorm * 6.5);
    float particle_mode = (sphere_style > 0.5 && sphere_style < 1.5) ? 1.0 : 0.0;
    float plasma_mode = sphere_style > 1.5 ? 1.0 : 0.0;
    float flow = 0.5 + 0.5 * sin(n3 * 9.0 + dnorm * 20.0 - t * (2.2 + sphere_state));
    swirl = mix(swirl, flow, particle_mode * 0.7);
    swirl = mix(swirl, 0.5 + 0.5 * sin(dnorm * 34.0 - t * 4.0 + n4 * 6.0), plasma_mode);
    float breath = 0.82 + 0.18 * sin(t * 1.6);
    float amplitude = (0.20 + 0.24 * sphere_state) * energy;
    float rim = pow(dnorm, 3.0);
    vec3 orb = state_col * (0.22 + swirl * 0.82 * breath + rim * 0.95) * (0.78 + amplitude);
    orb += vec3(1.0) * pow(max(0.0, dot(normalize(q + vec2(0.0001)), vec2(-0.55, 0.65))), 18.0) * (0.32 + 0.42 * sphere_state);
    orb += sphere_glow * (0.10 + 0.20 * swirl) * (0.35 + rim * 0.65);

    vec2 sgrid = q * 190.0 + vec2(t * 2.0, -t * 1.4);
    float spark = step(0.975 - 0.02 * sphere_state, hash21(floor(sgrid)))
                  * smoothstep(0.42, 0.0, length(fract(sgrid) - 0.5));
    vec2 pgrid = q * 260.0 + vec2(-t * 1.6, t * 1.9);
    spark = max(spark, particle_mode * step(0.93, hash21(floor(pgrid)))
                       * smoothstep(0.46, 0.0, length(fract(pgrid) - 0.5)));
    orb += state_col * spark * (0.55 + 1.8 * step(0.5, sphere_state) + particle_mode * 0.8);

    vec2 so = (shock_origin - vec2(0.5, 0.5)) * vec2(aspect, 1.0);
    float sdist = length(p - so);
    float age = time - shock_time;
    float wave = 0.0;
    if (age > 0.0 && age < 2.2) {
        float ring = smoothstep(0.045, 0.0, abs(sdist - age * 0.62));
        wave = ring * (1.0 - age / 2.2);
        orb += vec3(0.65, 1.0, 1.0) * wave * 0.9;
        col += vec3(0.35, 0.85, 1.0) * wave * 0.30;
    }

    float alpha = clamp(mask * sphere_alpha, 0.0, 1.0);
    col = mix(col, orb, alpha);
    col += state_col * exp(-pow((dist - radius * 1.16) * 15.0, 2.0)) * 0.20 * sphere_alpha;

    float vignette = smoothstep(1.28 * max(aspect, 1.0), 0.20, length(p));
    col *= 0.58 + 0.42 * vignette;
    col += (hash21(frag + fract(t)) - 0.5) * 0.02 * intensity;
    gl_FragColor = vec4(col, 1.0);
}
"""

AURORA_FALLBACK_FS = """
$HEADER$
uniform vec2 resolution;
uniform vec3 bg_color;
uniform vec3 color_a;
void main(void) {
    vec2 uv = vec2(frag_modelview_mat * gl_FragCoord).xy / max(resolution, vec2(1.0));
    vec3 col = mix(bg_color, color_a * 0.35, uv.y);
    gl_FragColor = vec4(col, 1.0);
}
"""


class AuroraBackground(Widget):
    """Fullscreen liquid aurora + living sphere rendered in one GLSL pass."""

    def __init__(self, palette: Dict[str, str], **kwargs: Any) -> None:
        self.canvas = RenderContext()
        super().__init__(**kwargs)
        self._palette = palette
        self.state_value = 0.0
        self.target_state = 0.0
        self.sphere_alpha = 1.0
        self.target_sphere_alpha = 1.0
        self.energy = 0.65
        self.intensity = 1.0
        self.fps = 40
        self.tilt = [0.0, 0.0]
        self.target_tilt = [0.0, 0.0]
        self.sphere_style = 0.0
        self.sphere_tint = [0.0, 0.0, 0.0]
        self.sphere_glow = [0.0, 0.0, 0.0]
        self.sphere_center = [0.5, 0.80]
        self.sphere_radius = 0.145
        self.shock_origin = [0.5, 0.5]
        self.shock_time = -10.0
        self._last_frame = 0.0
        self._fallback = False
        self._active = True
        self._event: Optional[Any] = None
        with self.canvas:
            Color(1, 1, 1, 1)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync, size=self._sync)
        shader = self.canvas.shader
        try:
            shader.fs = AURORA_FS
            if not shader.success:
                raise RuntimeError("fragment shader compilation failed")
            LOG.info("aurora GLSL shader compiled")
        except Exception as exc:  # noqa: BLE001
            LOG.error("aurora shader unavailable, using fallback", exc)
            self._fallback = True
            try:
                shader.fs = AURORA_FALLBACK_FS
            except Exception:  # noqa: BLE001
                pass
        self.set_palette(palette)
        self._event = Clock.schedule_interval(self._frame, 0)

    def stop(self) -> None:
        self._active = False
        if self._event is not None:
            with contextlib.suppress(Exception):
                self._event.cancel()
            self._event = None

    def _sync(self, *args: Any) -> None:
        self._rect.pos = self.pos
        self._rect.size = self.size

    def set_palette(self, palette: Dict[str, str]) -> None:
        self._palette = palette
        try:
            self.canvas["bg_color"] = _rgba(palette["bg"])[:3]
            self.canvas["color_a"] = _rgba(palette["accent"])[:3]
            self.canvas["color_b"] = _rgba(palette["accent2"])[:3]
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"palette uniform update failed: {exc}")

    def _frame(self, dt: float) -> None:
        if not self._active:
            return
        now = Clock.get_boottime()
        if now - self._last_frame < 1.0 / max(12, int(self.fps)):
            return
        self._last_frame = now
        self.state_value += (self.target_state - self.state_value) * 0.10
        self.sphere_alpha += (self.target_sphere_alpha - self.sphere_alpha) * 0.07
        for index in (0, 1):
            self.tilt[index] += (self.target_tilt[index] - self.tilt[index]) * 0.06
        try:
            self.canvas["time"] = now
            self.canvas["resolution"] = [float(self.width), float(self.height)]
            self.canvas["tilt"] = [float(v) for v in self.tilt]
            self.canvas["sphere_state"] = float(self.state_value)
            self.canvas["sphere_alpha"] = float(self.sphere_alpha)
            self.canvas["energy"] = float(self.energy)
            self.canvas["shock_origin"] = [float(v) for v in self.shock_origin]
            self.canvas["shock_time"] = float(self.shock_time)
            self.canvas["intensity"] = float(self.intensity)
            self.canvas["sphere_style"] = float(self.sphere_style)
            self.canvas["sphere_tint"] = [float(v) for v in self.sphere_tint]
            self.canvas["sphere_glow"] = [float(v) for v in self.sphere_glow]
            self.canvas["sphere_center"] = [float(v) for v in self.sphere_center]
            self.canvas["sphere_radius"] = float(self.sphere_radius)
            window_rc = Window.render_context
            self.canvas["projection_mat"] = window_rc["projection_mat"]
            self.canvas["modelview_mat"] = window_rc["modelview_mat"]
            self.canvas["frag_modelview_mat"] = window_rc["frag_modelview_mat"]
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"shader uniform update failed: {exc}")


class GlassPanel(BoxLayout):
    """Rounded translucent surface used by every dashboard panel."""

    def __init__(self, palette: Dict[str, str], alpha: float = 0.62, radius: int = 14,
                 border: bool = True, border_alpha: float = 0.30, **kwargs: Any) -> None:
        self.palette = palette
        self._radius = dp(radius)
        super().__init__(**kwargs)
        with self.canvas.before:
            self._bg = Color(*_rgba(palette["panel"], alpha))
            self._rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[self._radius])
            self._border = None
            if border:
                Color(*_rgba(palette["accent"], border_alpha))
                self._border = Line(width=1.0)
        self.bind(pos=self._sync, size=self._sync)
        self._sync()

    def _sync(self, *args: Any) -> None:
        self._rect.pos = self.pos
        self._rect.size = self.size
        if self._border is not None:
            self._border.rounded_rectangle = (
                self.x, self.y, self.width, self.height, self._radius
            )


BUTTON_STYLE = "glass"


# --------------------------------------------------------------------------
# UI sound: short feedback tones generated locally (no bundled audio files)
# --------------------------------------------------------------------------
class UISound:
    """Synthesizes click/toggle tones into the private cache and plays them."""

    SAMPLE_RATE = 22050
    NOTES = {
        "tap_glass": ((720.0, 0.055, 1.0),),
        "tap_rounded": ((900.0, 0.055, 1.0),),
        "tap_pill": ((1140.0, 0.055, 1.0),),
        "toggle_on": ((980.0, 0.060, 1.0),),
        "toggle_off": ((620.0, 0.060, 0.9),),
        "danger": ((310.0, 0.090, 0.9),),
        "test": ((660.0, 0.080, 0.8), (880.0, 0.080, 0.8), (1320.0, 0.110, 0.7)),
    }

    def __init__(self, config: Any, bridge: Optional[Any] = None) -> None:
        self.config = config
        self.bridge = bridge
        self._lock = threading.RLock()
        self._sounds: Dict[str, Any] = {}
        self._dir = PRIVATE_DIR / "cache" / "ui"
        self._kivy_failed = False
        self.last_error = ""
        self.plays = 0

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "ui_sound", True))

    @property
    def volume(self) -> float:
        return clamp(float(getattr(self.config, "ui_sound_volume", 0.35) or 0.0), 0.0, 1.0)

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self._kivy_failed and self.last_error:
            return "fallback · " + self.last_error[:90]
        return f"ready · volume {int(round(self.volume * 100))}%"

    def _tap_kind(self) -> str:
        return {"rounded": "tap_rounded", "pill": "tap_pill"}.get(BUTTON_STYLE, "tap_glass")

    def _write_wav(self, path: Path, notes: Sequence[Tuple[float, float, float]]) -> bool:
        frames = bytearray()
        for frequency, duration, gain in notes:
            count = max(1, int(self.SAMPLE_RATE * duration))
            fade_in = max(1, int(self.SAMPLE_RATE * 0.004))
            for index in range(count):
                t = index / self.SAMPLE_RATE
                envelope = min(1.0, index / fade_in) * math.exp(-4.2 * t / max(duration, 0.001))
                tone = math.sin(2.0 * math.pi * frequency * t)
                tone += 0.22 * math.sin(4.0 * math.pi * frequency * t)
                sample = int(32767 * 0.55 * gain * envelope * tone / 1.22)
                frames += struct.pack("<h", max(-32768, min(32767, sample)))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self.SAMPLE_RATE)
                handle.writeframes(bytes(frames))
            return True
        except (OSError, wave.Error) as exc:
            self.last_error = f"cannot write sound: {exc}"
            LOG.warning(self.last_error)
            return False

    def path_for(self, kind: str) -> Path:
        return self._dir / f"ui_{kind}.wav"

    def _ensure_file(self, kind: str) -> Optional[Path]:
        path = self.path_for(kind)
        with self._lock:
            try:
                if path.exists() and path.stat().st_size > 64:
                    return path
            except OSError:
                pass
            return path if self._write_wav(path, self.NOTES[kind]) else None

    def _load(self, kind: str) -> Any:
        path = self._ensure_file(kind)
        if path is None:
            raise RuntimeError(self.last_error or "sound file unavailable")
        from kivy.core.audio import SoundLoader

        sound = SoundLoader.load(str(path))
        if sound is None:
            raise RuntimeError("no audio provider accepted the generated WAV")
        return sound

    def play(self, kind: str = "tap") -> bool:
        if not self.enabled:
            return False
        volume = self.volume
        if volume <= 0.001:
            return False
        if kind == "tap":
            kind = self._tap_kind()
        if kind not in self.NOTES:
            kind = "tap_glass"
        if not self._kivy_failed:
            try:
                with self._lock:
                    sound = self._sounds.get(kind)
                    if sound is None:
                        sound = self._load(kind)
                        self._sounds[kind] = sound
                    sound.volume = volume
                    with contextlib.suppress(Exception):
                        sound.stop()
                    sound.play()
                self.plays += 1
                self.last_error = ""
                return True
            except Exception as exc:  # noqa: BLE001
                self._kivy_failed = True
                self.last_error = str(exc)
                LOG.warning(f"UI sound playback failed: {exc}")
        return self._system_click(volume)

    def _system_click(self, volume: float) -> bool:
        bridge = self.bridge
        if bridge is None:
            return False
        try:
            return bool(bridge.play_click(volume))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"system click failed: {exc}"
            return False

    def test(self) -> bool:
        return self.play("test")

    def warm_up(self) -> bool:
        with self._lock:
            return all(self._ensure_file(kind) is not None for kind in self.NOTES)

    def reset(self) -> None:
        """Forget cached players after the cache folder is cleared."""
        with self._lock:
            for sound in self._sounds.values():
                with contextlib.suppress(Exception):
                    sound.stop()
            self._sounds.clear()
            self._kivy_failed = False


_UI_SOUND: Optional[UISound] = None


def set_ui_sound_service(service: Optional[UISound]) -> None:
    global _UI_SOUND
    _UI_SOUND = service


def ui_sound_service() -> Optional[UISound]:
    return _UI_SOUND


class NeonButton(Button):
    def __init__(self, palette: Dict[str, str], tone: str = "accent", alpha: float = 0.20,
                 **kwargs: Any) -> None:
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_down", "")
        kwargs.setdefault("background_color", _rgba(palette.get(tone, palette["accent"]), alpha))
        kwargs.setdefault("color", _rgba(palette["text"], 0.96))
        kwargs.setdefault("font_name", FONT)
        kwargs.setdefault("font_size", sp(11))
        kwargs.setdefault("border", (0, 0, 0, 0))
        shrink = bool(kwargs.pop("shrink", True))
        raw_text = str(kwargs.get("text", ""))
        length = len(raw_text)
        display = raw_text
        try:
            numeric = float(kwargs["font_size"])
            width_hint = kwargs.get("width")
            has_fixed_width = isinstance(width_hint, (int, float))
            too_wide = (
                has_fixed_width
                and " · " in raw_text
                and length * numeric * 0.62 > float(width_hint) - dp(6)
            )
            if too_wide:
                display = raw_text.split(" · ", 1)[0]
            elif shrink and has_fixed_width:
                factor = 1.0
                if length > 18:
                    factor = 0.58
                elif length > 12:
                    factor = 0.70
                elif length > 9:
                    factor = 0.82
                if factor < 1.0:
                    kwargs["font_size"] = numeric * factor
        except (TypeError, ValueError):
            pass
        kwargs["text"] = "\n".join(shape_text(line) for line in display.split("\n"))
        self.palette = palette
        self._tone = tone
        self._style = BUTTON_STYLE
        super().__init__(**kwargs)
        self._fill_rgba = tuple(self.background_color)
        self.background_color = (0, 0, 0, 0)
        with self.canvas.before:
            self._fill_color = Color(*self._fill_rgba)
            self._fill_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(8)])
            self._border_color = Color(*_rgba(palette.get(tone, palette["accent"]), 0.42))
            self._border_line = Line(width=1.0)
        self.bind(
            pos=self._sync_shape, size=self._sync_shape,
            state=self._sync_shape, disabled=self._sync_shape,
            background_color=self._on_background_color,
        )
        self.sound_kind = "tap"
        self.bind(on_release=self._play_ui_sound)
        self._sync_shape()

    def _play_ui_sound(self, *_args: Any) -> None:
        if self.disabled:
            return
        kind = getattr(self, "sound_kind", "tap")
        if kind == "silent":
            return
        service = ui_sound_service()
        if service is not None:
            service.play("danger" if kind == "tap" and self._tone == "danger" else kind)

    def set_fill(self, rgba: Sequence[float]) -> None:
        self._fill_rgba = tuple(rgba)
        self._sync_shape()

    def _on_background_color(self, _widget: Any, value: Sequence[float]) -> None:
        if tuple(value) != (0.0, 0.0, 0.0, 0.0):
            self.set_fill(value)

    def _corner_radius(self) -> float:
        if self._style == "pill":
            return max(self.height / 2.0, dp(2))
        if self._style == "rounded":
            return dp(13)
        return dp(7)

    def _sync_shape(self, *_args: Any) -> None:
        radius = self._corner_radius()
        self._fill_rect.pos = self.pos
        self._fill_rect.size = self.size
        self._fill_rect.radius = [radius]
        self._border_line.rounded_rectangle = (self.x, self.y, self.width, self.height, radius)
        rgba = list(self._fill_rgba)
        if self.disabled:
            rgba[3] = rgba[3] * 0.45
        elif self.state == "down":
            rgba[3] = min(1.0, rgba[3] + 0.24)
        self._fill_color.rgba = rgba
        self._border_color.rgba = _rgba(
            self.palette.get(self._tone, self.palette["accent"]),
            0.62 if self.state == "down" else (0.34 if self._style == "glass" else 0.48),
        )


class Chip(NeonButton):
    def __init__(self, palette: Dict[str, str], **kwargs: Any) -> None:
        min_width = kwargs.pop("min_width", None)
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("height", dp(28))
        kwargs.setdefault("size_hint_x", None)
        kwargs.setdefault("font_size", sp(9.5))
        super().__init__(palette, shrink=False, **kwargs)
        width = len(self.text) * float(self.font_size) * 0.62 + dp(14)
        self.width = max(min_width if min_width is not None else dp(56), width)


class ToggleSwitch(Widget):
    """Compact neon toggle.

    The stock Kivy Switch draws an "ON"/"OFF" caption inside its own texture,
    which is clipped at the narrow widths used by the permission rows.
    """

    active = BooleanProperty(False)

    def __init__(self, palette: Dict[str, str], **kwargs: Any) -> None:
        kwargs.setdefault("size_hint", (None, 1))
        kwargs.setdefault("width", dp(46))
        super().__init__(**kwargs)
        self.palette = palette
        with self.canvas:
            self._track_color = Color(*_rgba(palette["dim"], 0.35))
            self._track = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(9)])
            self._knob_color = Color(*_rgba(palette["text"], 0.9))
            self._knob = Ellipse(pos=self.pos, size=self.size)
        self.bind(pos=self._sync, size=self._sync, active=self._sync)
        self._sync()

    def _sync(self, *_args: Any) -> None:
        height = min(self.height, dp(20))
        if height <= 0 or self.width <= 0:
            return
        self._track.size = (self.width, height)
        self._track.pos = (self.x, self.center_y - height / 2)
        self._track.radius = [height / 2]
        diameter = max(height - dp(4), dp(2))
        travel = max(self.width - diameter - dp(4), 0)
        knob_x = self.x + dp(2) + (travel if self.active else 0)
        self._knob.size = (diameter, diameter)
        self._knob.pos = (knob_x, self.center_y - diameter / 2)
        self._track_color.rgba = (
            _rgba(self.palette["accent"], 0.80) if self.active else _rgba(self.palette["dim"], 0.30)
        )
        self._knob_color.rgba = (
            _rgba(self.palette["bg"], 0.92) if self.active else _rgba(self.palette["text"], 0.95)
        )

    def on_touch_down(self, touch: Any) -> bool:
        if self.collide_point(*touch.pos):
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch: Any) -> bool:
        if touch.grab_current is self:
            touch.ungrab(self)
            if self.collide_point(*touch.pos):
                self.active = not self.active
                service = ui_sound_service()
                if service is not None:
                    service.play("toggle_on" if self.active else "toggle_off")
            return True
        return super().on_touch_up(touch)


@lru_cache(maxsize=8192)
def _word_width(word: str, font_size: float) -> float:
    from kivy.core.text import Label as CoreLabel

    probe = CoreLabel(font_name=FONT, font_size=font_size, text=shape_text(word), padding=0)
    probe.refresh()
    return float(probe.texture.size[0])


def _wrap_logical(text: str, width: float, font_size: float) -> str:
    """Wrap logical text, shaping each finished line on its own.

    Shaping a whole mixed Arabic/English paragraph before wrapping makes the
    bidi algorithm reorder across line breaks, scrambling line order. Wrapping
    the logical text first keeps every visual line internally consistent.
    """
    if width <= dp(20):
        return shape_text(text)
    space = _word_width(" ", font_size)
    lines: List[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        used = _word_width(line, font_size)
        for word in words[1:]:
            word_width = _word_width(word, font_size)
            if used + space + word_width <= width:
                line = f"{line} {word}"
                used += space + word_width
            else:
                lines.append(line)
                line = word
                used = word_width
        lines.append(line)
    return "\n".join(shape_text(line) for line in lines)


class WrapLabel(Label):
    """Label that wraps at its width and reports its own height."""

    def __init__(self, **kwargs: Any) -> None:
        logical = bool(kwargs.pop("logical", False))
        self._logical_text = str(kwargs.get("text", ""))
        if logical:
            kwargs["text"] = ""
        kwargs.setdefault("font_name", FONT)
        kwargs.setdefault("size_hint_y", None)
        kwargs.setdefault("halign", "left")
        kwargs.setdefault("valign", "top")
        kwargs.setdefault("markup", False)
        super().__init__(**kwargs)
        self.text_size = (self.width, None)
        self.bind(width=self._on_width, texture_size=self._on_texture)
        if logical:
            self.bind(width=self._rewrap_logical)
            self._rewrap_logical()

    def set_logical(self, text: str) -> None:
        self._logical_text = text
        self._rewrap_logical()

    def _rewrap_logical(self, *_args: Any) -> None:
        self.text = _wrap_logical(self._logical_text, max(self.width - dp(2), dp(20)), float(self.font_size))

    def _on_width(self, *args: Any) -> None:
        self.text_size = (max(self.width, dp(10)), None)

    def _on_texture(self, *args: Any) -> None:
        self.height = self.texture_size[1] + dp(2)


def _label(text: str, palette: Dict[str, str], size: float = 11, color: str = "text",
           bold: bool = False, halign: str = "left", **kwargs: Any) -> Label:
    kwargs.setdefault("size_hint_y", None)
    if "height" not in kwargs:
        kwargs["height"] = max(dp(13), sp(size) * 1.75)
    custom_text_size = "text_size" in kwargs
    label = Label(
        text=shape_text(text),
        font_name=FONT,
        font_size=sp(size),
        bold=bold,
        color=_rgba(palette[color]),
        halign=halign,
        valign="middle",
        **kwargs,
    )
    if not custom_text_size:
        label.bind(width=lambda widget, width: setattr(widget, "text_size", (max(1.0, width), None)))
        label.text_size = (max(1.0, label.width), None)
    return label


class BidiTextInput(TextInput):
    """Editable field that paints shaped Arabic while keeping logical text.

    Kivy's SDL2 text provider has no complex-script support, so Arabic appears
    as disconnected, reversed glyphs. Shaping is applied only to the painted
    string; `text` stays logical so search, routing and storage remain intact.
    Caret, tap-to-position and selection are mirrored for RTL lines.
    """

    auto_bidi = BooleanProperty(True)
    auto_direction = BooleanProperty(True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bind(text=self._on_bidi_changed, hint_text=self._on_bidi_changed)
        self._apply_direction()

    # -- painting --------------------------------------------------------
    def _display_text(self, text: str) -> str:
        if text and getattr(self, "auto_bidi", True) and contains_arabic(text):
            return shape_text(text)
        return text

    def _create_line_label(self, text: str, hint: bool = False):
        return super()._create_line_label(self._display_text(text), hint)

    def _get_text_width(self, text: str, tab_width: int, _label_cached: Any) -> float:
        if text and getattr(self, "auto_bidi", True) and contains_arabic(text):
            text = shape_text(text)
        return TextInput._get_text_width(self, text, tab_width, _label_cached)

    def _shaped_extent(self, text: str) -> float:
        if not text:
            return 0.0
        try:
            return float(
                TextInput._get_text_width(
                    self, shape_text(text), self.tab_width, self._label_cached
                )
            )
        except Exception:
            return 0.0

    def _line_is_rtl(self, line: str) -> bool:
        return bool(getattr(self, "auto_direction", True)) and text_is_rtl_dominant(line)

    def _rtl_offset(self, line: str, col: int) -> float:
        total = self._shaped_extent(line)
        return max(0.0, total - self._shaped_extent((line or "")[:col]))

    # -- caret and selection mapping -------------------------------------
    def cursor_offset(self) -> float:
        row, col = int(self.cursor_row), int(self.cursor_col)
        if 0 <= row < len(self._lines):
            line = self._lines[row]
            if self._line_is_rtl(line):
                return self._rtl_offset(line, col)
        return TextInput.cursor_offset(self)

    def get_cursor_from_xy(self, x: float, y: float) -> Tuple[int, int]:
        lines = self._lines
        if not lines:
            return 0, 0
        left, top, right, bottom = self.padding
        dy = self.line_height + self.line_spacing
        scroll_y = self.scroll_y / dy if self.scroll_y > 0 else 0
        row = int(
            min(
                max(
                    round(((self.top - top + scroll_y * dy) - y) / dy - 0.5),
                    0,
                ),
                len(lines) - 1,
            )
        )
        line = lines[row]
        if not self._line_is_rtl(line):
            return TextInput.get_cursor_from_xy(self, x, y)
        viewport_width = self.width - left - right
        row_width = self._get_row_width(row)
        xoff = 0.0
        if self.halign == "center":
            xoff = max(0.0, (viewport_width - row_width) / 2)
        elif self.halign == "right":
            xoff = max(0.0, viewport_width - row_width)
        target = x - self.x - left + self.scroll_x - xoff
        best_col, best_delta = 0, float("inf")
        for col in range(len(line) + 1):
            delta = abs(self._rtl_offset(line, col) - target)
            if delta < best_delta:
                best_col, best_delta = col, delta
        return best_col, row

    def _draw_selection(
        self, pos, size, line_num, selection_start, selection_end, lines,
        get_text_width, tab_width, label_cached, width_minus_padding,
        padding_left, padding_right, x, canvas_add, selection_color,
    ) -> None:
        line = lines[line_num] if line_num < len(lines) else ""
        if not self._line_is_rtl(line):
            return TextInput._draw_selection(
                self, pos, size, line_num, selection_start, selection_end, lines,
                get_text_width, tab_width, label_cached, width_minus_padding,
                padding_left, padding_right, x, canvas_add, selection_color,
            )
        start_col, start_row = selection_start
        end_col, end_row = selection_end
        if not start_row <= line_num <= end_row:
            return
        rect_x, rect_y = pos
        _, height = size
        total = self._shaped_extent(line)
        start_offset = (
            max(0.0, total - self._shaped_extent(line[:start_col]))
            if line_num == start_row else 0.0
        )
        end_offset = (
            max(0.0, total - self._shaped_extent(line[:end_col]))
            if line_num == end_row else total
        )
        beg_off, end_off = sorted((start_offset, end_offset))
        beg = min(max(rect_x - self.scroll_x + beg_off, x), x + width_minus_padding)
        end = min(max(rect_x - self.scroll_x + end_off, x), x + width_minus_padding)
        if beg == end:
            return
        canvas_add(Color(*selection_color, group="selection"))
        canvas_add(Rectangle(pos=(beg, rect_y), size=(end - beg, height), group="selection"))

    # -- alignment -------------------------------------------------------
    def _apply_direction(self, *_args: Any) -> None:
        if not getattr(self, "auto_direction", True):
            return
        probe = self.text or self.hint_text or ""
        wanted = "right" if text_is_rtl(probe) else "left"
        if self.halign != wanted:
            self.halign = wanted

    def _on_bidi_changed(self, *_args: Any) -> None:
        self._apply_direction()


def _shaped_input(palette: Dict[str, str], **kwargs: Any) -> BidiTextInput:
    if "hint_text" in kwargs:
        kwargs["hint_text"] = str(kwargs["hint_text"])  # logical; shaped at paint time
    kwargs.setdefault("font_name", FONT)
    kwargs.setdefault("font_size", sp(12))
    kwargs.setdefault("background_normal", "")
    kwargs.setdefault("background_active", "")
    kwargs.setdefault("background_color", _rgba(palette["panel"], 0.55))
    kwargs.setdefault("foreground_color", _rgba(palette["text"]))
    kwargs.setdefault("cursor_color", _rgba(palette["accent"]))
    kwargs.setdefault("hint_text_color", _rgba(palette["dim"], 0.9))
    kwargs.setdefault("padding", [dp(8), dp(8), dp(8), dp(8)])
    kwargs.setdefault("multiline", False)
    kwargs.setdefault("auto_bidi", True)
    kwargs.setdefault("auto_direction", True)
    return BidiTextInput(**kwargs)


class SphereTouch(Widget):
    """Invisible hit area over the shader sphere; triggers shockwave + haptics."""

    def __init__(self, app: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app = app

    def on_touch_down(self, touch: Any) -> bool:
        if not self.collide_point(*touch.pos):
            return False
        if self.app.current_screen != "chat":
            return False
        self.app.trigger_shock(touch.x, touch.y)
        return True


class BoshraModal(ModalView):
    """Glass modal with a scrollable body and a stable close control."""

    def __init__(self, app: Any, title: str, palette: Dict[str, str], **kwargs: Any) -> None:
        kwargs.setdefault("size_hint", (0.94, None))
        kwargs.setdefault("auto_dismiss", False)
        kwargs.setdefault("background_color", (0, 0, 0, 0))
        kwargs.setdefault("overlay_color", (0.0, 0.0, 0.0, 0.74))
        kwargs.setdefault("height", min(Window.height * 0.84, dp(700)))
        super().__init__(**kwargs)
        self.app = app
        self.palette = palette
        shell = GlassPanel(
            palette, orientation="vertical", alpha=0.97, radius=16,
            padding=dp(10), spacing=dp(8),
        )
        head = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(6))
        head.add_widget(_label(title, palette, 14, "accent", bold=True, halign="left"))
        close = NeonButton(palette, text="✕", size_hint=(None, 1), width=dp(34), font_size=sp(13))
        close.bind(on_release=lambda *_: self.dismiss())
        head.add_widget(close)
        shell.add_widget(head)
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False, bar_width=dp(2))
        self.body = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(7), padding=dp(2))
        self.body.bind(minimum_height=self.body.setter("height"))
        scroll.add_widget(self.body)
        shell.add_widget(scroll)
        self.add_widget(shell)

    def open(self, *args: Any, **kwargs: Any) -> Any:
        registry = getattr(self.app, "modals", None)
        if isinstance(registry, list) and self not in registry:
            registry.append(self)
        return super().open(*args, **kwargs)

    def dismiss(self, *args: Any, **kwargs: Any) -> Any:
        registry = getattr(self.app, "modals", None)
        if isinstance(registry, list):
            with contextlib.suppress(ValueError):
                registry.remove(self)
        return super().dismiss(*args, **kwargs)

    def text(self, value: str, size: float = 11, color: str = "text", **kwargs: Any) -> Label:
        label = WrapLabel(text=shape_text(value), font_size=sp(size), color=_rgba(self.palette[color]), **kwargs)
        self.body.add_widget(label)
        return label

    def button(self, text: str, callback: Callable[..., Any], tone: str = "accent") -> Button:
        button = NeonButton(self.palette, text=shape_text(text), tone=tone, size_hint_y=None, height=dp(36))
        button.bind(on_release=lambda *_: callback())
        self.body.add_widget(button)
        return button


class ConfirmModal(BoshraModal):
    def __init__(self, app: Any, question: str, detail: str, palette: Dict[str, str],
                 callback: Callable[[bool], None]) -> None:
        super().__init__(app, T("Confirmation", "تأكيد"), palette,
                         height=min(Window.height * 0.54, dp(440)))
        self.callback = callback
        self.text(question, 11.5)
        if detail:
            self.text(detail[:1200], 9.5, "dim")
        row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        no_btn = NeonButton(palette, text=T("Cancel", "إلغاء"), tone="danger")
        yes_btn = NeonButton(palette, text=T("Confirm", "تأكيد"), tone="accent")
        no_btn.bind(on_release=lambda *_: self._finish(False))
        yes_btn.bind(on_release=lambda *_: self._finish(True))
        row.add_widget(no_btn)
        row.add_widget(yes_btn)
        self.body.add_widget(row)

    def _finish(self, answer: bool) -> None:
        self.dismiss()
        try:
            self.callback(bool(answer))
        except Exception as exc:  # noqa: BLE001
            LOG.error("confirm callback failed", exc)


PLAN_STEP_MARKS = {"done": "✓", "failed": "✕", "running": "…", "pending": "○"}


class PlanModal(BoshraModal):
    """Visible plan mode: propose, inspect, run and watch each tool step."""

    def __init__(self, app: Any, goal: str = "") -> None:
        super().__init__(app, T("Plan mode", "وضع الخطة"), app.palette,
                         height=min(Window.height * 0.86, dp(660)))
        self.plan: Optional[Dict[str, Any]] = None
        self.goal_input = _shaped_input(
            self.palette, text=goal, hint_text=T(
                "A goal, e.g. list tasks; add a task: call mom; note: weekly review",
                "هدف، مثل: اعرض المهام؛ أضف مهمة: اتصال بأمي؛ ملاحظة: مراجعة أسبوعية",
            ),
            size_hint_y=None, height=dp(38), multiline=False,
        )
        self.body.add_widget(self.goal_input)
        actions = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
        build_btn = NeonButton(self.palette, text=T("Build", "ابنِ"), font_size=sp(8.5))
        build_btn.bind(on_release=lambda *_: self.app.ui_guard(self.build_plan))
        run_btn = NeonButton(self.palette, text=T("Run", "نفّذ"), font_size=sp(8.5))
        run_btn.bind(on_release=lambda *_: self.app.ui_guard(self.run_plan))
        cancel_btn = NeonButton(self.palette, text=T("Cancel", "إلغاء"), tone="danger", font_size=sp(8.5))
        cancel_btn.bind(on_release=lambda *_: self.cancel_plan())
        actions.add_widget(build_btn)
        actions.add_widget(run_btn)
        actions.add_widget(cancel_btn)
        self.body.add_widget(actions)
        self.status = self.text(T("Describe a goal, then build the plan.", "صِف هدفاً ثم ابنِ الخطة."), 8.8, "dim")
        self.steps_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        self.steps_box.bind(minimum_height=self.steps_box.setter("height"))
        self.body.add_widget(self.steps_box)
        self.result = self.text("", 9.2, "text")

    def build_plan(self) -> None:
        goal = (self.goal_input.text or "").strip()
        if not goal:
            self.status.text = shape_text(T("Write the goal first.", "اكتب الهدف أولاً."))
            return
        outcome = self.app.core.planner.propose(goal)
        if not outcome.get("ok"):
            self.status.text = shape_text(str(outcome.get("error") or "Failed"))
            return
        self.plan = outcome.get("plan")
        steps = (self.plan or {}).get("steps") or []
        self.status.text = shape_text(T(
            f"Plan #{self.plan['id']} · {len(steps)} steps · ready to run",
            f"خطة #{self.plan['id']} · {len(steps)} خطوات · جاهزة للتنفيذ",
        ))
        self.result.text = ""
        self._render_steps()

    def _render_steps(self) -> None:
        if not self.plan:
            return
        fresh = self.app.core.planner.get(int(self.plan["id"])) or self.plan
        self.plan = fresh
        self.steps_box.clear_widgets()
        for index, step in enumerate(fresh.get("steps") or [], 1):
            status = str(step.get("status", "pending"))
            tone = {"done": "good", "failed": "danger", "running": "accent"}.get(status, "dim")
            label = WrapLabel(
                text=shape_text(f"{PLAN_STEP_MARKS.get(status, '•')} {index}. {step.get('title', '')}"),
                font_size=sp(9.4), color=_rgba(self.palette[tone]),
                size_hint_y=None,
            )
            label.bind(width=lambda widget, width: setattr(widget, "text_size", (width, None)))
            label.bind(texture_size=lambda widget, size: setattr(widget, "height", max(dp(18), size[1] + dp(2))))
            self.steps_box.add_widget(label)
            detail = str(step.get("detail") or "")
            if detail:
                sub = WrapLabel(
                    text=shape_text(detail[:260]), font_size=sp(7.8),
                    color=_rgba(self.palette["dim"]), size_hint_y=None,
                )
                sub.bind(width=lambda widget, width: setattr(widget, "text_size", (width, None)))
                sub.bind(texture_size=lambda widget, size: setattr(widget, "height", max(dp(14), size[1] + dp(2))))
                self.steps_box.add_widget(sub)

    def run_plan(self) -> None:
        if self.plan is None:
            self.build_plan()
        if self.plan is None:
            return
        plan_id = int(self.plan["id"])
        self.status.text = shape_text(T("Running the plan…", "جارٍ تنفيذ الخطة…"))
        self.result.text = ""

        def work() -> Dict[str, Any]:
            return self.app.core.planner.run(
                plan_id, on_step=lambda _index, _step: self.app._ui_after(self._render_steps)
            )

        self.app.run_bg(work, self._finished)

    def _finished(self, outcome: Dict[str, Any]) -> None:
        self._render_steps()
        summary = str(outcome.get("summary") or outcome.get("error") or "")
        status = str(outcome.get("status") or ("done" if outcome.get("ok") else "partial"))
        self.status.text = shape_text(T(f"Plan finished · {status}", f"انتهت الخطة · {status}"))
        self.result.text = shape_text(summary[:1800])
        chat = self.app.ensure_screen("chat")
        if chat is not None and summary:
            with contextlib.suppress(Exception):
                chat.add_message("assistant", summary)

    def cancel_plan(self) -> None:
        with contextlib.suppress(Exception):
            self.app.core.orchestrator.stop_event.set()
        self.status.text = shape_text(T("Cancellation requested.", "طُلب الإلغاء."))


class RecallModal(BoshraModal):
    """Semantic recall across every local data source, with citations."""

    def __init__(self, app: Any, query: str = "") -> None:
        super().__init__(app, T("Recall everything", "استرجاع شامل"), app.palette,
                         height=min(Window.height * 0.84, dp(620)))
        self.last_hits: List[Dict[str, Any]] = []
        self.query_input = _shaped_input(
            self.palette, text=query, hint_text=T(
                "Search chats, notes, tasks, memory, events, follow-ups and vault",
                "ابحث في المحادثات والملاحظات والمهام والذاكرة والمواعيد والمتابعات والخزنة",
            ),
            size_hint_y=None, height=dp(38), multiline=False,
        )
        self.body.add_widget(self.query_input)
        actions = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
        search_btn = NeonButton(self.palette, text=T("Search", "ابحث"), font_size=sp(8.5))
        search_btn.bind(on_release=lambda *_: self.app.ui_guard(self.search))
        index_btn = NeonButton(self.palette, text=T("Reindex", "أعد الفهرسة"), font_size=sp(8.5))
        index_btn.bind(on_release=lambda *_: self.app.ui_guard(self.reindex))
        chat_btn = NeonButton(self.palette, text=T("To chat", "إلى المحادثة"), font_size=sp(8.5))
        chat_btn.bind(on_release=lambda *_: self.send_to_chat())
        actions.add_widget(search_btn)
        actions.add_widget(index_btn)
        actions.add_widget(chat_btn)
        self.body.add_widget(actions)
        self.status = self.text(
            T("The index covers everything stored on this device.",
              "يشمل الفهرس كل ما هو مخزّن على هذا الجهاز."),
            8.6, "dim",
        )
        self.results_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        self.results_box.bind(minimum_height=self.results_box.setter("height"))
        self.body.add_widget(self.results_box)

    def search(self) -> None:
        query = (self.query_input.text or "").strip()
        if not query:
            self.status.text = shape_text(T("Type what you are looking for.", "اكتب ما تبحث عنه."))
            return
        self.status.text = shape_text(T("Searching locally…", "جارٍ البحث محلياً…"))
        index = self.app.core.recall

        def work() -> List[Dict[str, Any]]:
            index.reindex_if_stale()
            return index.search(query, k=14)

        self.app.run_bg(work, self._render_hits)

    def reindex(self) -> None:
        self.status.text = shape_text(T("Rebuilding the index…", "جارٍ إعادة بناء الفهرس…"))

        def work() -> Dict[str, int]:
            return self.app.core.recall.reindex()

        def done(counts: Dict[str, int]) -> None:
            total = sum((counts or {}).values())
            self.status.text = shape_text(T(f"Index rebuilt · {total} items",
                                            f"أُعيد بناء الفهرس · {total} عنصر"))

        self.app.run_bg(work, done)

    def _render_hits(self, hits: List[Dict[str, Any]]) -> None:
        self.last_hits = list(hits or [])
        self.results_box.clear_widgets()
        if not hits:
            self.status.text = shape_text(T("No local matches.", "لا توجد نتائج محلية."))
            return
        self.status.text = shape_text(T(f"{len(hits)} matches", f"{len(hits)} نتيجة"))
        for hit in hits:
            english, arabic = RECALL_KINDS.get(hit["kind"], (hit["kind"], hit["kind"]))
            title = _label(
                f"{T(english, arabic)} · {hit.get('title', '')} · {hit.get('score', 0)}",
                self.palette, 8.4, "accent", bold=True,
            )
            snippet = WrapLabel(
                text=shape_text(str(hit.get("snippet", ""))), font_size=sp(8.2),
                color=_rgba(self.palette["text"]), size_hint_y=None,
            )
            snippet.bind(width=lambda widget, width: setattr(widget, "text_size", (width, None)))
            snippet.bind(texture_size=lambda widget, size: setattr(widget, "height", max(dp(16), size[1] + dp(2))))
            block = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None, alpha=0.34,
                radius=8, padding=[dp(5), dp(3)], spacing=dp(1), border_alpha=0.12,
            )
            block.bind(minimum_height=block.setter("height"))
            block.add_widget(title)
            block.add_widget(snippet)
            self.results_box.add_widget(block)

    def send_to_chat(self) -> None:
        if not self.last_hits:
            self.status.text = shape_text(T("Search first.", "ابحث أولاً."))
            return
        lines = [T(f"Recall for “{self.query_input.text.strip()}”",
                   f"استرجاع عن «{self.query_input.text.strip()}»")]
        for hit in self.last_hits[:10]:
            english, arabic = RECALL_KINDS.get(hit["kind"], (hit["kind"], hit["kind"]))
            lines.append(f"• [{T(english, arabic)} · {hit.get('title', '')}] {hit.get('snippet', '')}")
        body = "\n".join(lines)
        chat = self.app.ensure_screen("chat")
        self.app.show_screen("chat")
        if chat is not None:
            with contextlib.suppress(Exception):
                chat.add_message("assistant", body)
        self.dismiss()


class BackupModal(BoshraModal):
    """Encrypted backups: create, list and stage a restore."""

    def __init__(self, app: Any) -> None:
        super().__init__(app, T("Encrypted backups", "النسخ الاحتياطية المشفرة"), app.palette,
                         height=min(Window.height * 0.86, dp(660)))
        self.text(T(
            "Backups include the database, settings, skills and vault index. "
            "Leave the password empty to seal with this device's key.",
            "تشمل النسخ قاعدة البيانات والإعدادات والمهارات وفهرس الخزنة. "
            "اترك كلمة المرور فارغة للختم بمفتاح هذا الجهاز.",
        ), 8.6, "dim")
        self.password = _shaped_input(
            self.palette, password=True, hint_text=T("Optional password", "كلمة مرور اختيارية"),
            size_hint_y=None, height=dp(36),
        )
        self.body.add_widget(self.password)
        actions = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
        create_btn = NeonButton(self.palette, text=T("Create backup", "أنشئ نسخة"), font_size=sp(8.4))
        create_btn.bind(on_release=lambda *_: self.app.ui_guard(self.create))
        refresh_btn = NeonButton(self.palette, text=T("Refresh", "تحديث"), font_size=sp(8.4))
        refresh_btn.bind(on_release=lambda *_: self.app.ui_guard(self.refresh))
        actions.add_widget(create_btn)
        actions.add_widget(refresh_btn)
        self.body.add_widget(actions)
        self.status = self.text("", 8.8, "dim")
        self.list_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        self.list_box.bind(minimum_height=self.list_box.setter("height"))
        self.body.add_widget(self.list_box)
        self.refresh()

    def create(self) -> None:
        password = (self.password.text or "").strip()
        label = "portable" if password else "device"
        self.status.text = shape_text(T("Creating an encrypted snapshot…", "جارٍ إنشاء نسخة مشفرة…"))

        def work() -> Dict[str, Any]:
            return self.app.core.backups.create(password=password, label=label)

        def done(outcome: Dict[str, Any]) -> None:
            if outcome.get("ok"):
                self.status.text = shape_text(str(outcome.get("message", "Backup created")))
                self.password.text = ""
                self.refresh()
            else:
                self.status.text = shape_text(str(outcome.get("error") or "Backup failed"))

        self.app.run_bg(work, done)

    def refresh(self) -> None:
        self.list_box.clear_widgets()
        entries = self.app.core.backups.list_backups()
        if not entries:
            self.list_box.add_widget(_label(
                T("No backups yet.", "لا توجد نسخ بعد."), self.palette, 8.6, "dim",
            ))
            return
        for entry in entries[:12]:
            row = GlassPanel(
                self.palette, orientation="horizontal", size_hint_y=None, height=dp(40),
                alpha=0.34, radius=8, padding=[dp(5), dp(2)], spacing=dp(3), border_alpha=0.12,
            )
            size_kb = int(entry.get("size") or 0) // 1024
            row.add_widget(WrapLabel(
                text=shape_text(f"{Path(str(entry.get('path', ''))).name[:34]}\n"
                                f"{entry.get('mode')} · {size_kb} KB · {str(entry.get('created', ''))[5:16]}"),
                font_size=sp(7.8), color=_rgba(self.palette["dim"]),
            ))
            restore_btn = NeonButton(self.palette, text=T("Restore", "استعادة"),
                                     size_hint=(None, 1), width=dp(58), font_size=sp(7.6))
            restore_btn.bind(on_release=lambda *_args, path=str(entry.get("path", "")):
                             self.app.ui_guard(lambda: self.restore(path)))
            row.add_widget(restore_btn)
            self.list_box.add_widget(row)

    def restore(self, path: str) -> None:
        password = (self.password.text or "").strip()

        def confirmed(answer: bool) -> None:
            if not answer:
                return
            self.status.text = shape_text(T("Verifying and staging the backup…",
                                            "جارٍ التحقق من النسخة وتجهيزها…"))

            def work() -> Dict[str, Any]:
                return self.app.core.backups.restore(path, password=password)

            def done(outcome: Dict[str, Any]) -> None:
                if outcome.get("ok"):
                    self.status.text = shape_text(str(outcome.get("message", "Ready to restore")))
                    self.app.toast(str(outcome.get("message", "")), 6)
                else:
                    self.status.text = shape_text(str(outcome.get("error") or "Restore failed"))

            self.app.run_bg(work, done)

        self.app.confirm_async(
            T("Restore this backup?", "استعادة هذه النسخة؟"),
            T("Current data is kept aside and Boshra restarts to apply it.",
              "ستُحفظ البيانات الحالية جانباً وستُعاد بشرى التشغيل لتطبيقها."),
            confirmed,
        )


class DigestModal(BoshraModal):
    def __init__(self, app: Any, body: str, kind: str = "manual") -> None:
        title = {
            "morning": T("Morning brief", "موجز الصباح"),
            "evening": T("Evening review", "مراجعة المساء"),
        }.get(kind, T("Chief-of-staff brief", "موجز المساعد"))
        super().__init__(app, title, app.palette, height=min(Window.height * 0.82, dp(620)))
        self.body_text = body
        label = WrapLabel(text=shape_text(body), font_size=sp(9.4), color=_rgba(self.palette["text"]),
                          size_hint_y=None)
        label.bind(width=lambda widget, width: setattr(widget, "text_size", (width, None)))
        label.bind(texture_size=lambda widget, height: setattr(widget, "height", max(dp(20), height[1] + dp(4))))
        self.body.add_widget(label)
        row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(6))
        add_btn = NeonButton(self.palette, text=T("Add to chat", "أضف إلى المحادثة"), font_size=sp(8.6))
        add_btn.bind(on_release=lambda *_: self._to_chat())
        close_btn = NeonButton(self.palette, text=T("Close", "إغلاق"), font_size=sp(8.6))
        close_btn.bind(on_release=lambda *_: self.dismiss())
        row.add_widget(add_btn)
        row.add_widget(close_btn)
        self.body.add_widget(row)

    def _to_chat(self) -> None:
        chat = self.app.ensure_screen("chat")
        self.app.show_screen("chat")
        if chat is not None:
            with contextlib.suppress(Exception):
                chat.add_message("assistant", self.body_text)
        self.dismiss()


class MiniAssistantModal(BoshraModal):
    """Compact in-app assistant used by the floating orb fallback."""

    def __init__(self, app: Any) -> None:
        super().__init__(app, T("Mini assistant", "المساعد المصغّر"), app.palette,
                         height=min(Window.height * 0.58, dp(420)))
        self._busy = False
        self.reply = WrapLabel(
            text=T("Ask a short question. The full chat stays one tap away.",
                   "اسأل سؤالاً قصيراً. المحادثة الكاملة على بعد لمسة."),
            logical=True, font_size=sp(9.6), color=_rgba(self.palette["text"]),
            size_hint_y=None,
        )
        self.reply.bind(width=lambda widget, width: setattr(widget, "text_size", (width, None)))
        self.reply.bind(texture_size=lambda widget, size: setattr(widget, "height", max(dp(20), size[1] + dp(4))))
        self.body.add_widget(self.reply)
        self.input = _shaped_input(
            self.palette, hint_text=T("Type a quick request…", "اكتب طلباً سريعاً…"),
            size_hint_y=None, height=dp(38),
        )
        self.input.bind(on_text_validate=lambda *_: self.send())
        self.body.add_widget(self.input)
        row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
        send_btn = NeonButton(self.palette, text=T("Send", "أرسل"), font_size=sp(8.6))
        send_btn.bind(on_release=lambda *_: self.send())
        full_btn = NeonButton(self.palette, text=T("Full chat", "المحادثة"), font_size=sp(8.6))
        full_btn.bind(on_release=lambda *_: self._full_chat())
        row.add_widget(send_btn)
        row.add_widget(full_btn)
        self.body.add_widget(row)

    def send(self) -> None:
        text = (self.input.text or "").strip()
        if not text or self._busy:
            return
        self._busy = True
        self.input.text = ""
        self.reply.set_logical(T("Thinking locally…", "أفكر محلياً…"))

        def work() -> str:
            return self.app.core.process_text(text)

        def done(result: str) -> None:
            self._busy = False
            self.reply.set_logical(str(result or ""))

        self.app.run_bg(work, done)

    def _full_chat(self) -> None:
        self.dismiss()
        self.app.show_screen("chat")
        chat = self.app.ensure_screen("chat")
        if chat is not None:
            with contextlib.suppress(Exception):
                chat.load_history()


class DraggableOrb(Widget):
    """Small draggable assistant bubble; a tap opens the mini assistant."""

    def __init__(self, app: Any, **kwargs: Any) -> None:
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("size", (dp(54), dp(54)))
        super().__init__(**kwargs)
        self.app = app
        self._touch_origin: Optional[Tuple[float, float, float, float]] = None
        self._moved = False
        primary, glow = sphere_colors(app.core.config.sphere_color)
        with self.canvas:
            self._halo_color = Color(primary[0], primary[1], primary[2], 0.24)
            self._halo = Ellipse()
            self._body_color = Color(primary[0], primary[1], primary[2], 0.92)
            self._body = Ellipse()
            self._ring_color = Color(glow[0], glow[1], glow[2], 0.92)
            self._ring = Line(width=dp(1.1))
        self._label = Label(
            text="◎", font_name=FONT, font_size=sp(19), bold=True,
            color=(0.02, 0.04, 0.07, 0.96), halign="center", valign="middle",
        )
        self.bind(pos=self._sync_canvas, size=self._sync_canvas)
        self._sync_canvas()

    def apply_sphere_colors(self, value: str) -> None:
        primary, glow = sphere_colors(value)
        self._halo_color.rgba = (primary[0], primary[1], primary[2], 0.24)
        self._body_color.rgba = (primary[0], primary[1], primary[2], 0.92)
        self._ring_color.rgba = (glow[0], glow[1], glow[2], 0.92)

    def _sync_canvas(self, *_args: Any) -> None:
        x, y = self.pos
        width, height = self.size
        pad = dp(5)
        self._halo.pos = (x - pad, y - pad)
        self._halo.size = (width + pad * 2, height + pad * 2)
        self._body.pos = (x, y)
        self._body.size = (width, height)
        self._ring.circle = (x + width / 2, y + height / 2, min(width, height) / 2 - dp(1))
        self._label.pos = self.pos
        self._label.size = self.size

    def on_touch_down(self, touch: Any) -> bool:
        if not self.collide_point(*touch.pos):
            return False
        self._touch_origin = (touch.x, touch.y, self.x, self.y)
        self._moved = False
        return True

    def on_touch_move(self, touch: Any) -> bool:
        if self._touch_origin is None:
            return False
        start_x, start_y, base_x, base_y = self._touch_origin
        if not self._moved and abs(touch.x - start_x) + abs(touch.y - start_y) < dp(7):
            return True
        self._moved = True
        parent = self.parent
        max_x = (parent.width if parent else Window.width) - self.width
        max_y = (parent.height if parent else Window.height) - self.height
        self.pos = (clamp(base_x + touch.x - start_x, 0, max(max_x, 0)),
                    clamp(base_y + touch.y - start_y, 0, max(max_y, 0)))
        return True

    def on_touch_up(self, touch: Any) -> bool:
        if self._touch_origin is None:
            return False
        self._touch_origin = None
        if not self._moved:
            self.app.ui_guard(self.app.open_mini_assistant)
        else:
            with contextlib.suppress(Exception):
                self.app.core.store.state_set("float:pos", f"{int(self.x)},{int(self.y)}")
        return True


class ScreenBase(BoxLayout):
    key = ""
    backdrop = True

    def __init__(self, app: Any) -> None:
        self.app = app
        self.core = app.core
        self.palette = app.palette
        super().__init__(orientation="vertical", spacing=dp(5), padding=[dp(4), dp(4), dp(4), dp(4)])
        self._backdrop_rect = None
        if self.backdrop:
            with self.canvas.before:
                Color(*_rgba(self.palette["bg"], 0.66))
                self._backdrop_rect = Rectangle(pos=self.pos, size=self.size)
            self.bind(pos=self._sync_backdrop, size=self._sync_backdrop)

    def _sync_backdrop(self, *_args: Any) -> None:
        if self._backdrop_rect is not None:
            self._backdrop_rect.pos = self.pos
            self._backdrop_rect.size = self.size

    def build_header(self, title: str, actions: Optional[Sequence[Tuple[str, Callable[[], Any]]]] = None) -> BoxLayout:
        actions = list(actions or [])
        panel = GlassPanel(
            self.palette, orientation="vertical", size_hint_y=None,
            height=dp(58) if actions else dp(30),
            alpha=0.72, radius=10, padding=[dp(6), dp(2), dp(6), dp(2)], spacing=dp(1),
        )
        panel.add_widget(_label(title, self.palette, 12.0, "accent", bold=True, halign="left", shorten=True))
        row = BoxLayout(size_hint_y=None, height=dp(30), spacing=dp(4))
        for text, callback in actions:
            label = text.split(" · ", 1)[0] if _UI_LANG == "bilingual" else text
            button = NeonButton(
                self.palette, text=label, shrink=False,
                size_hint_x=1, font_size=sp(7.8), alpha=0.16,
            )
            button.bind(on_release=lambda *_args, cb=callback: self.app.ui_guard(cb))
            row.add_widget(button)
        panel.add_widget(row)
        self.add_widget(panel)
        return row

    def list_area(self) -> Tuple[ScrollView, BoxLayout]:
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False, bar_width=dp(2))
        box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(5), padding=dp(1))
        box.bind(minimum_height=box.setter("height"))
        scroll.add_widget(box)
        self.add_widget(scroll)
        return scroll, box

    def toolbar(self, height: int = 40) -> BoxLayout:
        row = BoxLayout(size_hint_y=None, height=dp(height), spacing=dp(4))
        self.add_widget(row)
        return row

    def refresh(self) -> None:
        return


class ResultModal(BoshraModal):
    def __init__(self, app: Any, title: str, text: str) -> None:
        super().__init__(app, title, app.palette)
        self.text(text[:12000], 10.5)


class ChatScreen(ScreenBase):
    key = "chat"
    backdrop = False
    sphere_band = 0.34

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.padding = [dp(6), dp(2), dp(6), dp(4)]
        self.spacing = 0
        self._pending = ""
        self._current_body: Optional[WrapLabel] = None
        self._flush_trigger = Clock.create_trigger(self._flush_tokens, 0.05)
        self._draft_trigger = Clock.create_trigger(self._save_draft, 0.4)
        self._message_rows: List[Dict[str, Any]] = []
        self._last_day = ""
        self._loading_history = False
        self.add_widget(Widget(size_hint_y=self.sphere_band))
        panel = GlassPanel(
            self.palette, orientation="vertical", size_hint_y=1.0 - self.sphere_band,
            alpha=0.60, radius=16,
            padding=[dp(7), dp(4), dp(7), dp(6)], spacing=dp(3),
        )
        self.add_widget(panel)

        status_row = BoxLayout(size_hint_y=None, height=dp(26), spacing=dp(4))
        self.status = _label(
            "", self.palette, 8.2, "dim", halign="left", shorten=True,
            height=dp(24), text_size=(dp(120), dp(24)),
        )
        self.status.bind(width=lambda widget, width: setattr(widget, "text_size", (width, dp(24))))
        status_row.add_widget(self.status)
        self.ctx_label = _label(
            "ctx --", self.palette, 7.2, "dim", halign="right", shorten=True,
            size_hint=(None, 1), width=dp(60), height=dp(22),
            text_size=(dp(60), dp(22)),
        )
        status_row.add_widget(self.ctx_label)
        new_btn = NeonButton(
            self.palette, text=T1("New", "جديد"), size_hint=(None, 1), width=dp(40), font_size=sp(8.2),
        )
        new_btn.bind(on_release=lambda *_: self.app.ui_guard(self.app.new_conversation))
        status_row.add_widget(new_btn)
        commands_btn = NeonButton(
            self.palette, text=T1("Cmds", "أوامر"), size_hint=(None, 1), width=dp(44), font_size=sp(8.2),
        )
        commands_btn.bind(on_release=lambda *_: self.app.ui_guard(self.open_commands))
        status_row.add_widget(commands_btn)
        export_btn = NeonButton(
            self.palette, text=T1("Save", "حفظ"), size_hint=(None, 1), width=dp(40), font_size=sp(8.2),
        )
        export_btn.bind(on_release=lambda *_: self.app.ui_guard(self.export_conversation))
        status_row.add_widget(export_btn)
        panel.add_widget(status_row)

        self.log_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False, bar_width=dp(2))
        self.log_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(5), padding=[dp(1), dp(2)])
        self.log_box.bind(minimum_height=self.log_box.setter("height"))
        self.log_scroll.add_widget(self.log_box)
        panel.add_widget(self.log_scroll)

        chips_scroll = ScrollView(
            size_hint_y=None, height=dp(31), do_scroll_x=True, do_scroll_y=False,
            bar_width=0, scroll_type=["content"],
        )
        chips = BoxLayout(size_hint=(None, 1), spacing=dp(3))
        chips.bind(minimum_width=chips.setter("width"))
        quick = [
            (T1("My day", "يومي"), "daily briefing"),
            (T1("Plan", "خطة"), "__plan__"),
            (T1("Recall", "استرجاع"), "__recall__"),
            (T1("Brief", "موجز"), "__brief__"),
            (T1("Float", "عائم"), "__float__"),
            (T1("More", "المزيد"), "__more__"),
        ]
        for label_text, command in quick:
            chip = Chip(self.palette, text=label_text, min_width=dp(44), font_size=sp(9.0))
            chip.bind(on_release=lambda *_args, cmd=command: self._send_quick(cmd))
            chips.add_widget(chip)
        chips_scroll.add_widget(chips)
        panel.add_widget(chips_scroll)

        input_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(3))
        self.input = _shaped_input(
            self.palette, hint_text=L("ask_placeholder"), size_hint=(1, 1),
        )
        self.input.bind(on_text_validate=lambda *_: self.app.ui_guard(self.send_current))
        self.input.bind(focus=lambda _widget, focused: self.scroll_bottom() if focused else None)
        self.input.bind(text=self._on_composer_changed)
        input_row.add_widget(self.input)
        self.mic_btn = NeonButton(self.palette, text="Mic", size_hint=(None, 1), width=dp(42), font_size=sp(8.5))
        self.mic_btn.bind(on_release=lambda *_: self.app.ui_guard(lambda: self.app.toggle_voice("command")))
        input_row.add_widget(self.mic_btn)
        self.wake_btn = NeonButton(self.palette, text="Wake", size_hint=(None, 1), width=dp(45), font_size=sp(8.5))
        self.wake_btn.bind(on_release=lambda *_: self.app.ui_guard(lambda: self.app.toggle_voice("wake")))
        input_row.add_widget(self.wake_btn)
        self.stop_btn = NeonButton(
            self.palette, text="Stop", tone="danger", size_hint=(None, 1), width=dp(42), font_size=sp(8.5),
        )
        self.stop_btn.bind(on_release=lambda *_: self.app.ui_guard(self.app.stop_all))
        input_row.add_widget(self.stop_btn)
        self.send_btn = NeonButton(
            self.palette, text="›", size_hint=(None, 1), width=dp(38), font_size=sp(18), bold=True,
        )
        self.send_btn.bind(on_release=lambda *_: self.app.ui_guard(self.send_current))
        input_row.add_widget(self.send_btn)
        panel.add_widget(input_row)

        self.send_btn.disabled = True
        self.load_history()
        self.refresh_status()
        self.restore_draft()

    # -- drafts and composer ---------------------------------------------
    def _draft_key(self) -> str:
        return f"draft:{int(self.core.config.last_conversation_id or 0)}"

    def draft_text(self) -> str:
        return self.core.store.state_get(self._draft_key(), "")

    def restore_draft(self, focus: bool = False) -> None:
        if (self.input.text or "").strip():
            return
        draft = self.draft_text()
        if draft:
            self.input.text = draft
            self.input.cursor = (len(draft), 0)
            self.input.halign = "right" if text_is_rtl(draft) else "left"
            if focus:
                self.input.focus = True
        self._on_composer_changed()

    def _save_draft(self, *_args: Any) -> None:
        try:
            self.core.store.state_set(self._draft_key(), self.input.text or "")
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"draft save failed: {exc}")

    def _on_composer_changed(self, *_args: Any) -> None:
        text = self.input.text or ""
        self.send_btn.disabled = not bool(text.strip())
        if text.strip():
            language = "AR" if text_is_rtl(text) else "EN"
            words = len(text.split())
            self.status.text = shape_text(f"{language} · {len(text)} chars · {words} words")
        else:
            self.refresh_status()
        if not self._loading_history:
            self._draft_trigger()

    def reset_timeline(self) -> None:
        self._message_rows.clear()
        self._last_day = ""
        self._pending = ""
        self._current_body = None

    # -- timeline --------------------------------------------------------
    def load_history(self) -> None:
        self.reset_timeline()
        self.log_box.clear_widgets()
        self._loading_history = True
        try:
            for message in self.core.orchestrator.history(limit=30):
                self.add_message(
                    str(message.get("role", "assistant")),
                    str(message.get("content", "")),
                    ts=str(message.get("ts", "")),
                    notify=False,
                )
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"chat history failed: {exc}")
        finally:
            self._loading_history = False
        self.scroll_bottom()

    def _day_label(self, ts: str) -> str:
        moment = parse_iso(ts) or datetime.now()
        today = datetime.now().date()
        if moment.date() == today:
            return T("Today", "اليوم")
        if moment.date() == today - timedelta(days=1):
            return T("Yesterday", "أمس")
        return moment.strftime("%Y-%m-%d")

    def _add_day_separator(self, ts: str) -> None:
        day = (parse_iso(ts) or datetime.now()).date().isoformat()
        if day == self._last_day:
            return
        self._last_day = day
        self.log_box.add_widget(_label(
            "· " + self._day_label(ts) + " ·", self.palette, 7.4, "dim", halign="center",
        ))

    def _action_button(self, parent: Any, label: str, callback: Callable[[], Any],
                       width: float = 44) -> None:
        button = NeonButton(
            self.palette, text=label, size_hint=(None, 1), width=dp(width),
            font_size=sp(7.2), alpha=0.16,
        )
        button.bind(on_release=lambda *_: self.app.ui_guard(callback))
        parent.add_widget(button)

    def add_message(self, role: str, text: str, streaming: bool = False,
                    ts: str = "", notify: bool = True) -> WrapLabel:
        ts = ts or now_iso()
        self._add_day_separator(ts)
        row = BoxLayout(size_hint_y=None, spacing=dp(3))
        row.bind(minimum_height=row.setter("height"))
        is_user = role == "user"
        bubble = GlassPanel(
            self.palette, orientation="vertical", size_hint=(0.88, None),
            alpha=0.72 if is_user else 0.50, radius=11,
            padding=[dp(7), dp(4), dp(7), dp(5)], spacing=dp(1),
            border_alpha=0.22,
        )
        bubble.bind(minimum_height=bubble.setter("height"))
        moment = parse_iso(ts)
        clock_text = moment.strftime("%H:%M") if moment else ""
        heading = (T("YOU", "أنت") if is_user else "BOSHRA") + (f" · {clock_text}" if clock_text else "")
        bubble.add_widget(_label(
            heading, self.palette, 7.5, "dim" if is_user else "accent", bold=True,
        ))
        body = WrapLabel(
            text=text,
            logical=True,
            font_size=sp(11.2),
            color=_rgba(self.palette["text"]),
        )
        bubble.add_widget(body)
        actions = BoxLayout(size_hint_y=None, height=dp(20), spacing=dp(3))
        if is_user:
            self._action_button(actions, T1("Edit", "تعديل"), lambda: self.edit_message(body), 44)
            self._action_button(actions, T1("Copy", "نسخ"), lambda: self.copy_reply(body), 44)
            actions.add_widget(Widget())
        else:
            self._action_button(actions, T1("Speak", "صوت"), lambda: self.speak_reply(body), 48)
            self._action_button(actions, T1("Copy", "نسخ"), lambda: self.copy_reply(body), 44)
            self._action_button(actions, T1("Pin", "تثبيت"), lambda: self.pin_to_memory(body), 40)
            self._action_button(actions, T1("TXT", "نص"), lambda: self.save_reply(body, "txt"), 40)
            self._action_button(actions, T1("PY", "كود"), lambda: self.save_reply(body, "py"), 38)
            self._action_button(actions, T1("Again", "إعادة"), lambda: self.app.regenerate_last(), 48)
            actions.add_widget(Widget())
        bubble.add_widget(actions)
        rtl_shell = bool(self.core.config.rtl_layout) and _UI_LANG == "ar"
        if is_user == rtl_shell:
            row.add_widget(bubble)
            row.add_widget(Widget(size_hint_x=0.12))
        else:
            row.add_widget(Widget(size_hint_x=0.12))
            row.add_widget(bubble)
        self.log_box.add_widget(row)
        self._message_rows.append(
            {"row": row, "bubble": bubble, "body": body, "role": role, "ts": ts}
        )
        if notify:
            self.scroll_bottom()
        return body

    def remove_last_assistant(self) -> bool:
        for entry in reversed(self._message_rows):
            if entry["role"] != "assistant":
                continue
            self._message_rows.remove(entry)
            with contextlib.suppress(Exception):
                self.log_box.remove_widget(entry["row"])
            return True
        return False

    def search_entries(self, query: str) -> List[Dict[str, Any]]:
        needle = normalize_arabic((query or "").lower()).strip()
        if not needle:
            return []
        hits: List[Dict[str, Any]] = []
        for entry in self._message_rows:
            text = str(getattr(entry["body"], "_logical_text", "") or "")
            if needle in normalize_arabic(text.lower()):
                hits.append(entry)
        return hits

    def scroll_to_entry(self, entry: Dict[str, Any]) -> None:
        row = entry.get("row")
        if row is None or row.parent is None:
            return
        with contextlib.suppress(Exception):
            self.log_scroll.scroll_to(row, padding=dp(12), animate=True)
        with contextlib.suppress(Exception):
            row.opacity = 0.55
            Clock.schedule_once(lambda _dt: setattr(row, "opacity", 1.0), 0.45)

    def scroll_bottom(self) -> None:
        Clock.schedule_once(lambda _dt: setattr(self.log_scroll, "scroll_y", 0), 0.03)

    def begin_assistant(self) -> None:
        self._pending = ""
        self._current_body = self.add_message("assistant", "", streaming=True)

    def append_token(self, token: str) -> None:
        self._pending += token
        self._flush_trigger()

    def _flush_tokens(self, *_args: Any) -> None:
        if self._current_body is not None:
            self._current_body.set_logical(self._pending)
            self.scroll_bottom()

    def finish_assistant(self, final_text: str) -> None:
        self._pending = final_text
        if self._current_body is not None:
            self._current_body.set_logical(final_text)
        self._current_body = None
        self.scroll_bottom()

    def set_partial(self, text: str) -> None:
        self.status.text = shape_text(("mic · " if not contains_arabic(text) else "mic · ") + text[:70])

    def refresh_status(self) -> None:
        try:
            engine = self.core.llm.engine_name
            voice_state = "voice ready" if (self.core.voice and self.core.voice.available) else "voice off"
            tts_state = "tts on" if self.core.tts.available else "tts off"
            mode = str(self.core.config.online_mode).upper()
            self.status.text = shape_text(f"{engine} · {voice_state} · {tts_state} · net:{mode}")
            if getattr(self, "ctx_label", None) is not None and self.core.compactor is not None:
                usage = self.core.compactor.usage()
                self.ctx_label.text = shape_text(T1(
                    f"ctx {usage['percent']}%", f"سياق {usage['percent']}%",
                ))
                self.ctx_label.color = _rgba(
                    self.palette["danger"] if usage["percent"] >= 85 else self.palette["dim"]
                )
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"status refresh: {exc}")

    def _send_quick(self, command: str) -> None:
        if command == "__history__":
            ConversationBrowserModal(self.app).open()
            return
        if command == "__find__":
            ConversationSearchModal(self.app, self).open()
            return
        if command == "__plan__":
            self.app.show_plan_modal()
            return
        if command == "__recall__":
            self.app.show_recall_modal()
            return
        if command == "__brief__":
            self.app.show_latest_digest()
            return
        if command == "__float__":
            self.app.toggle_float()
            return
        if command == "__more__":
            self.open_commands()
            return
        if command == "daily briefing" and _UI_LANG == "ar":
            command = "موجز اليوم"
        self.app.send_text(command)

    def send_current(self) -> None:
        text = (self.input.text or "").strip()
        if not text:
            return
        with contextlib.suppress(Exception):
            self._draft_trigger.cancel()
        self.input.text = ""
        with contextlib.suppress(Exception):
            self.core.store.state_set(self._draft_key(), "")
        self.app.send_text(text)

    def speak_reply(self, body: WrapLabel) -> None:
        text = str(getattr(body, "_logical_text", "") or "").strip()
        if not text:
            self.app.toast(T("This reply is still streaming.", "هذا الرد قيد التوليد."))
            return
        self.app.speak_text(text)

    def edit_message(self, body: WrapLabel) -> None:
        text = str(getattr(body, "_logical_text", "") or "").strip()
        if text:
            self.app.edit_in_composer(text)

    def copy_reply(self, body: WrapLabel) -> None:
        text = str(getattr(body, "_logical_text", "") or "").strip()
        if not text:
            return
        copied = self.core.bridge.clipboard_set(text)
        self.app.toast(T("Copied to clipboard.", "نُسخ إلى الحافظة.") if copied else
                       T("Clipboard unavailable.", "الحافظة غير متاحة."))

    def pin_to_memory(self, body: WrapLabel) -> None:
        text = str(getattr(body, "_logical_text", "") or "").strip()
        if not text:
            self.app.toast(T("This reply is still streaming.", "هذا الرد قيد التوليد."))
            return
        outcome = self.core.memory.add("pinned", text[:2000], source="chat", tags="pinned")
        if outcome.get("ok"):
            self.app.toast(T("Pinned to local memory.", "ثُبّت في الذاكرة المحلية."), 2.4)
        else:
            self.app.toast(str(outcome.get("error") or "Memory refused"), 4)

    def save_reply(self, body: WrapLabel, extension: str) -> None:
        text = str(getattr(body, "_logical_text", "") or "").strip()
        if not text:
            self.app.toast(T("Nothing to save yet.", "لا يوجد نص لحفظه بعد."))
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if extension == "py":
            blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, re.S)
            payload = "\n\n".join(block.strip() for block in blocks if block.strip())
            if not payload:
                payload = "# Boshra AI reply\n# " + "\n# ".join(text.splitlines()) + "\n"
        else:
            payload = text + "\n"
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPORT_DIR / f"boshra_reply_{stamp}.{extension}"
        if atomic_write_text(path, payload):
            with contextlib.suppress(Exception):
                self.core.store.audit("chat", "export-reply", path.name, "low")
            self.app.toast(T(f"Saved to {path}", f"حُفظ في {path}"), 5)
        else:
            self.app.toast(T("Could not write the reply file.", "تعذر كتابة ملف الرد."))

    def export_conversation(self) -> None:
        try:
            messages = self.core.orchestrator.history(limit=300)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"history export failed: {exc}")
            messages = []
        if not messages:
            self.app.toast(T("The conversation is empty.", "المحادثة فارغة."))
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lines = [f"{APP_NAME} {APP_VERSION} · {BUILD_STAMP}", ""]
        for message in messages:
            role = str(message.get("role", "assistant")).upper()
            lines.append(f"[{role}] {message.get('ts', '')}".rstrip())
            lines.append(str(message.get("content", "")))
            lines.append("")
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = EXPORT_DIR / f"boshra_chat_{stamp}.txt"
        if atomic_write_text(path, "\n".join(lines)):
            self.app.toast(T(f"Conversation saved to {path}", f"حُفظت المحادثة في {path}"), 6)
        else:
            self.app.toast(T("Could not write the conversation file.", "تعذر كتابة ملف المحادثة."))

    def open_commands(self) -> None:
        CommandListModal(self.app, T("All commands", "كل الأوامر"), self.palette, self.core).open()


SKILL_TABS = [
    ("", "All", "الكل"),
    ("productivity", "Tasks", "مهام"),
    ("documents", "Docs", "مستندات"),
    ("system", "System", "نظام"),
    ("intelligence", "Assist", "ذكاء"),
    ("development", "Dev", "تطوير"),
    ("online", "Online", "شبكة"),
    ("custom", "Custom", "مخصص"),
]


class SkillsScreen(ScreenBase):
    key = "skills"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.category = ""
        self.build_header(T("Skill Studio", "استوديو المهارات"), [
            (T("Builder", "باني"), self.open_builder),
            (T("Import", "استيراد"), self.import_skill),
            (T("Rescan", "فحص"), self.rescan),
        ])
        search_row = self.toolbar(38)
        self.search = _shaped_input(self.palette, hint_text=T("Search skills, commands, permissions", "ابحث في المهارات أو الأوامر"), size_hint=(1, 1))
        self.search.bind(on_text_validate=lambda *_: self.refresh())
        search_row.add_widget(self.search)
        go = NeonButton(self.palette, text=T("Find", "بحث"), size_hint=(None, 1), width=dp(48), font_size=sp(9))
        go.bind(on_release=lambda *_: self.refresh())
        search_row.add_widget(go)
        tabs_scroll = ScrollView(size_hint_y=None, height=dp(30), do_scroll_x=True, do_scroll_y=False, bar_width=0)
        tabs = BoxLayout(size_hint=(None, 1), spacing=dp(3))
        tabs.bind(minimum_width=tabs.setter("width"))
        self.tab_buttons: Dict[str, Chip] = {}
        for key, en, ar in SKILL_TABS:
            chip = Chip(self.palette, text=T(en, ar))
            chip.bind(on_release=lambda *_args, cat=key: self.set_category(cat))
            tabs.add_widget(chip)
            self.tab_buttons[key] = chip
        tabs_scroll.add_widget(tabs)
        self.add_widget(tabs_scroll)
        self.count_label = _label("", self.palette, 8.5, "dim")
        self.add_widget(self.count_label)
        self.list_scroll, self.list_box = self.list_area()
        self.refresh()

    def set_category(self, category: str) -> None:
        self.category = category
        for key, chip in self.tab_buttons.items():
            chip.background_color = _rgba(
                self.palette["accent"], 0.34 if key == category else 0.14
            )
        self.refresh()

    def refresh(self) -> None:
        try:
            skills = self.core.skills.list_skills(self.search.text if hasattr(self, "search") else "", self.category)
        except Exception as exc:  # noqa: BLE001
            LOG.error("skills list failed", exc)
            skills = []
        self.list_box.clear_widgets()
        enabled = sum(1 for skill in skills if skill["enabled"])
        self.count_label.text = shape_text(
            f"{len(skills)} skills · {enabled} enabled · {len(self.core.skills.invalid)} invalid"
        )
        if not skills:
            self.list_box.add_widget(WrapLabel(
                text=shape_text(T("No skills match this filter.", "لا توجد مهارات مطابقة.")),
                font_size=sp(10), color=_rgba(self.palette["dim"]),
            ))
            return
        for skill in skills:
            panel = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None,
                alpha=0.52, radius=10, padding=[dp(7), dp(4), dp(5), dp(4)], spacing=dp(2),
                border_alpha=0.20,
            )
            panel.bind(minimum_height=panel.setter("height"))
            top = BoxLayout(size_hint_y=None, height=dp(21), spacing=dp(3))
            status_dot = "●" if skill["enabled"] else "○"
            name = _label(
                f"{status_dot} {skill['name']}", self.palette, 10.5,
                "accent" if skill["enabled"] else "dim", bold=True,
            )
            top.add_widget(name)
            open_btn = NeonButton(
                self.palette, text=T("Open", "فتح"), size_hint=(None, 1), width=dp(44), font_size=sp(8),
            )
            open_btn.bind(on_release=lambda *_args, sid=skill["id"]: self.open_detail(sid))
            top.add_widget(open_btn)
            panel.add_widget(top)
            meta = (
                f"{skill['id']} · {skill['category']} · risk {skill['risk']} · "
                + ("offline" if skill["offline"] else "online")
                + ("" if skill["health"] == "healthy" else f" · {skill['health']}")
            )
            panel.add_widget(_label(meta, self.palette, 7.5, "dim"))
            description = str(skill.get("description", ""))
            if description:
                panel.add_widget(WrapLabel(
                    text=shape_text(description[:180]),
                    font_size=sp(9), color=_rgba(self.palette["text"], 0.82),
                ))
            self.list_box.add_widget(panel)

    def rescan(self) -> None:
        def work() -> Any:
            return self.core.skills.discover()
        self.app.run_bg(work, lambda result: (self.app.toast(T("Skills rescanned", "تم فحص المهارات")), self.refresh()))

    def open_detail(self, skill_id: str) -> None:
        SkillDetailModal(self.app, skill_id, self.refresh).open()

    def open_builder(self) -> None:
        SkillBuilderModal(self.app, self.refresh).open()

    def import_skill(self) -> None:
        self.app.run_bg(self.app.pick_path, self._after_pick)

    def _after_pick(self, path: Optional[str]) -> None:
        if not path:
            self.app.toast(T("No file selected (or picker unavailable on this platform).", "لم يتم اختيار ملف أو أن منتقي الملفات غير متاح."))
            return
        if not str(path).endswith(".boshra-skill") and not str(path).endswith(".zip"):
            self.app.toast(T("Select a .boshra-skill package", "اختر حزمة .boshra-skill"))
            return

        def work() -> Any:
            return self.core.skills.inspect_package(path)

        self.app.run_bg(work, lambda inspection: self._show_import(path, inspection))

    def _show_import(self, path: str, inspection: Any) -> None:
        if not inspection or not inspection.get("ok"):
            self.app.toast(str((inspection or {}).get("error", "Invalid package")))
            return
        modal = BoshraModal(self.app, T("Import skill package", "استيراد حزمة مهارة"), self.palette)
        manifest = inspection["manifest"]
        modal.text(manifest["name"], 14, "accent", bold=True)
        modal.text(
            f"{manifest['id']} · v{manifest['version']} · {manifest['category']}\n"
            f"risk: {manifest['risk_level']} · files: {inspection['files']} · "
            f"python entry: {'yes' if inspection.get('has_python') else 'no'}",
            9, "dim",
        )
        modal.text(manifest.get("description", ""), 10)
        modal.text(
            T(
                "Packages are treated as untrusted. Python entry points run in a separate process with an allow-list, not a security sandbox.",
                "تُعامل الحزم كغير موثوقة. يعمل إدخال بايثون في عملية منفصلة بقائمة سماح، وهذا ليس صندوقاً عازلاً أمنياً.",
            ),
            8.5, "danger",
        )

        def install() -> None:
            def work() -> Any:
                return self.core.skills.install_package(path)

            def done(result: Any) -> None:
                self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
                modal.dismiss()
                self.refresh()
            self.app.run_bg(work, done)
        modal.button(T("Install (disabled until reviewed)", "تثبيت (معطل حتى المراجعة)"), install)
        modal.open()


class SkillDetailModal(BoshraModal):
    def __init__(self, app: Any, skill_id: str, on_change: Callable[[], Any]) -> None:
        self.skill_id = skill_id
        self.on_change = on_change
        skill = app.core.skills.get(skill_id) or {}
        super().__init__(app, str(skill.get("name", skill_id)), app.palette)
        self.render()

    def render(self) -> None:
        self.body.clear_widgets()
        skill = self.app.core.skills.get(self.skill_id)
        if not skill:
            self.text(T("Skill not found.", "المهارة غير موجودة."), 11, "danger")
            return
        self.text(
            f"{skill['id']} · v{skill['version']} · {skill['category']}",
            8.5, "dim",
        )
        status = T("enabled", "ممكّنة") if skill["enabled"] else T("disabled", "معطّلة")
        health = skill["health"]
        self.text(
            f"{status} · {health} · {T('risk', 'الخطورة')}: {skill['risk']} · "
            + (T("offline", "دون اتصال") if skill["offline"] else T("requires internet", "يتطلب الإنترنت")),
            9, "good" if skill["enabled"] and health == "healthy" else "danger",
        )
        self.text(skill.get("description", ""), 10)
        if skill.get("builtin"):
            self.text(T("Built-in skill: permissions are enforced per tool call and can be revoked in Privacy.", "مهارة مدمجة: تُفرض الأذونات عند كل أداة ويمكن سحبها من الخصوصية."), 8.5, "dim")

        actions = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(4))
        toggle_text = T("Disable", "تعطيل") if skill["enabled"] else T("Enable", "تمكين")
        toggle = NeonButton(self.palette, text=toggle_text, tone="danger" if skill["enabled"] else "accent")
        toggle.bind(on_release=lambda *_: self._toggle())
        actions.add_widget(toggle)
        test = NeonButton(self.palette, text=T("Test", "اختبار"))
        test.bind(on_release=lambda *_: self._test())
        actions.add_widget(test)
        export_btn = NeonButton(self.palette, text=T("Export", "تصدير"))
        export_btn.bind(on_release=lambda *_: self._export())
        actions.add_widget(export_btn)
        self.body.add_widget(actions)

        second = BoxLayout(size_hint_y=None, height=dp(32), spacing=dp(4))
        if not skill.get("builtin"):
            uninstall = NeonButton(self.palette, text=T("Uninstall", "إزالة"), tone="danger")
            uninstall.bind(on_release=lambda *_: self._uninstall())
            second.add_widget(uninstall)
            rollback = NeonButton(self.palette, text=T("Rollback", "تراجع"))
            rollback.bind(on_release=lambda *_: self._rollback())
            second.add_widget(rollback)
        reset = NeonButton(self.palette, text=T("Reset config", "تصفير الإعدادات"))
        reset.bind(on_release=lambda *_: self._reset())
        second.add_widget(reset)
        self.body.add_widget(second)

        self.text(T("Permissions", "الأذونات"), 11, "accent", bold=True)
        grants = self.app.core.skills.grants(self.skill_id)
        permissions = list(skill.get("permissions", []))
        optional = list(skill.get("optional_permissions", []))
        if not permissions and not optional:
            self.text(T("No device permissions requested.", "لا تطلب أي أذونات جهاز."), 9, "dim")
        for permission in permissions + optional:
            row = BoxLayout(size_hint_y=None, height=dp(26), spacing=dp(4))
            risk, _why, label = PERMISSION_CATALOG.get(permission, ("low", "", permission))
            row.add_widget(_label(f"{label} · {permission} · {risk}", self.palette, 8.3, "text"))
            if skill.get("builtin"):
                row.add_widget(_label(T("built-in", "مدمج"), self.palette, 8, "good", halign="right"))
            else:
                switch = ToggleSwitch(self.palette, active=bool(grants.get(permission)), width=dp(46))
                switch.bind(active=lambda _w, value, perm=permission: self._permission(perm, bool(value)))
                row.add_widget(switch)
            self.body.add_widget(row)

        self.text(T("Commands", "الأوامر"), 11, "accent", bold=True)
        commands = skill.get("commands", [])
        if not commands:
            self.text(T("No declarative commands listed.", "لا توجد أوامر تعريفية."), 9, "dim")
        for command in commands:
            self.text(f"› {command.get('name', '?')} — {command.get('description', '')}", 9)

        self.text(T("Recent history", "السجل الأخير"), 11, "accent", bold=True)
        try:
            history = self.app.core.skills.history(self.skill_id, 6)
        except Exception:  # noqa: BLE001
            history = []
        if not history:
            self.text(T("No executions yet.", "لا يوجد تنفيذ بعد."), 8.5, "dim")
        for item in history:
            mark = "✓" if item.get("success") else "✗"
            self.text(f"{mark} {human_ts(item.get('ts', ''))} · {item.get('action', '')} · {item.get('message', '')[:90]}", 8.2, "dim")

    def _toggle(self) -> None:
        skill = self.app.core.skills.get(self.skill_id)
        if not skill:
            return

        def work() -> Any:
            if skill["enabled"]:
                return self.app.core.skills.disable(self.skill_id)
            return self.app.core.skills.enable(self.skill_id)

        def done(result: Any) -> None:
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
            self.on_change()
            self.render()
        self.app.run_bg(work, done)

    def _permission(self, permission: str, granted: bool) -> None:
        def work() -> Any:
            skills = self.app.core.skills
            return skills.grant(self.skill_id, permission) if granted else skills.revoke(self.skill_id, permission)
        self.app.run_bg(work, lambda _result: self.render())

    def _test(self) -> None:
        def work() -> Any:
            return self.app.core.skills.self_test(self.skill_id)

        def done(result: Any) -> None:
            if not result:
                return
            checks = result.get("checks", [])
            body = "\n".join(f"{'✓' if c.get('status') != 'fail' else '✗'} {c.get('name')} {c.get('detail', '')}" for c in checks)
            self.text(T("Self-test", "الاختبار الذاتي") + f": {'ok' if result.get('ok') else 'issues'}\n{body}", 9, "good" if result.get("ok") else "danger")
        self.app.run_bg(work, done)

    def _export(self) -> None:
        def work() -> Any:
            return self.app.core.skills.export_skill(self.skill_id)

        def done(result: Any) -> None:
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
        self.app.run_bg(work, done)

    def _uninstall(self) -> None:
        def confirmed(answer: bool) -> None:
            if not answer:
                return

            def work() -> Any:
                return self.app.core.skills.uninstall(self.skill_id)

            def done(result: Any) -> None:
                self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
                self.dismiss()
                self.on_change()
            self.app.run_bg(work, done)
        self.app.confirm_async(T("Uninstall this skill?", "إزالة هذه المهارة؟"), self.skill_id, confirmed)

    def _rollback(self) -> None:
        def work() -> Any:
            return self.app.core.skills.rollback(self.skill_id)

        def done(result: Any) -> None:
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
            self.on_change()
            self.render()
        self.app.run_bg(work, done)

    def _reset(self) -> None:
        def work() -> Any:
            return self.app.core.skills.reset_config(self.skill_id)

        def done(result: Any) -> None:
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
            self.render()
        self.app.run_bg(work, done)


class SkillBuilderModal(BoshraModal):
    def __init__(self, app: Any, on_change: Callable[[], Any]) -> None:
        super().__init__(app, T("Skill Builder", "باني المهارات"), app.palette)
        self.on_change = on_change
        self._proposal: Optional[Dict[str, Any]] = None
        self.text(
            T("Describe the capability. Boshra composes allow-listed tools; it never writes unrestricted code.",
              "صف القدرة المطلوبة. ستُركّب بشرى أدوات مسموحة فقط، ولا تولّد كوداً حراً."),
            9, "dim",
        )
        self.description = _shaped_input(
            self.palette, multiline=True,
            font_size=sp(10.5),
            hint_text=T("Example: summarize my study documents and save a report", "مثال: لخّص مستنداتي واكتب تقريراً"),
            size_hint_y=None, height=dp(74), padding=dp(7),
        )
        self.body.add_widget(self.description)
        propose = NeonButton(self.palette, text=T("Propose manifest", "اقترح البيان"), size_hint_y=None, height=dp(36))
        propose.bind(on_release=lambda *_: self._propose())
        self.body.add_widget(propose)
        self.output = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(4))
        self.output.bind(minimum_height=self.output.setter("height"))
        self.body.add_widget(self.output)

    def _propose(self) -> None:
        text = (self.description.text or "").strip()
        self.output.clear_widgets()

        def work() -> Any:
            return self.app.core.skills.propose_skill(text)

        def done(result: Any) -> None:
            if not result or not result.get("ok"):
                self.app.toast(str((result or {}).get("error", "Proposal failed")))
                return
            self._proposal = result["manifest"]
            manifest_json = json.dumps(result["manifest"], ensure_ascii=False, indent=2)
            self.output.add_widget(WrapLabel(
                text=manifest_json, font_size=sp(8), color=_rgba(self.palette["dim"]),
            ))
            tests = "\n".join(f"• {item['name']}: {item['status']}" for item in result.get("tests", []))
            self.output.add_widget(WrapLabel(
                text=shape_text(T("Checks", "الفحوصات") + "\n" + tests),
                font_size=sp(9), color=_rgba(self.palette["good"]),
            ))
            install = NeonButton(
                self.palette, text=T("Approve & install", "موافقة وتثبيت"),
                size_hint_y=None, height=dp(36),
            )
            install.bind(on_release=lambda *_: self._install())
            self.output.add_widget(install)
        self.app.run_bg(work, done)

    def _install(self) -> None:
        if not self._proposal:
            return

        def work() -> Any:
            return self.app.core.skills.install_generated(self._proposal)

        def done(result: Any) -> None:
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
            if (result or {}).get("ok"):
                self.dismiss()
                self.on_change()
        self.app.run_bg(work, done)


MEMORY_CATEGORY_AR = {
    "session": "الجلسة", "conversation": "المحادثة", "profile": "الملف",
    "episodic": "الوقائع", "semantic": "الدلالي", "procedural": "الإجراءات", "project": "المشاريع",
}


class MemoryScreen(ScreenBase):
    key = "memory"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Memory Center", "مركز الذاكرة"), [
            (T("Add", "إضافة"), self.open_add),
            (T("Export", "تصدير"), self.export_memory),
            (T("Import", "استيراد"), self.import_memory),
        ])
        row = self.toolbar(38)
        self.search = _shaped_input(self.palette, hint_text=T("Search memories", "ابحث في الذاكرة"), size_hint=(1, 1))
        self.search.bind(on_text_validate=lambda *_: self.refresh())
        row.add_widget(self.search)
        self.category = Spinner(
            text="all", values=["all"] + list(MEMORY_CATEGORIES), size_hint=(None, 1), width=dp(76),
            font_name=FONT, font_size=sp(8.5),
            background_normal="", background_color=_rgba(self.palette["accent"], 0.2),
            color=_rgba(self.palette["text"]),
        )
        self.category.bind(text=lambda *_: self.refresh())
        row.add_widget(self.category)
        self.list_scroll, self.list_box = self.list_area()
        self.refresh()

    def refresh(self) -> None:
        query = (self.search.text or "").strip() if hasattr(self, "search") else ""
        category = None if self.category.text in ("all", "") else self.category.text
        try:
            if query:
                rows = self.core.memory.search(query, k=60, category=category)
            else:
                rows = self.core.memory.list(category=category, limit=300)
        except Exception as exc:  # noqa: BLE001
            LOG.error("memory list failed", exc)
            rows = []
        self.list_box.clear_widgets()
        self.list_box.add_widget(_label(f"{len(rows)} memories", self.palette, 8.5, "dim"))
        if not rows:
            self.list_box.add_widget(WrapLabel(
                text=shape_text(T("Nothing stored yet. Say 'remember ...' or add a memory.", "لا يوجد مخزون بعد. قل «تذكر...» أو أضف ذكرى.")),
                font_size=sp(10), color=_rgba(self.palette["dim"]),
            ))
            return
        for memory in rows:
            panel = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None, alpha=0.50,
                radius=9, padding=[dp(7), dp(4), dp(5), dp(4)], spacing=dp(2), border_alpha=0.18,
            )
            panel.bind(minimum_height=panel.setter("height"))
            top = BoxLayout(size_hint_y=None, height=dp(20), spacing=dp(3))
            category_label = memory["category"]
            if _UI_LANG != "en":
                category_label += " · " + MEMORY_CATEGORY_AR.get(category_label.split(" · ")[0], "")
            top.add_widget(_label(category_label, self.palette, 8, "accent", bold=True))
            delete_btn = NeonButton(
                self.palette, text=T("Delete", "حذف"), tone="danger",
                size_hint=(None, 1), width=dp(48), font_size=sp(7.5),
            )
            delete_btn.bind(on_release=lambda *_args, mid=memory["id"]: self._delete(mid))
            top.add_widget(delete_btn)
            panel.add_widget(top)
            content = str(memory.get("content", ""))
            panel.add_widget(WrapLabel(
                text=shape_text(content[:400] + ("…" if len(content) > 400 else "")),
                font_size=sp(9.5), color=_rgba(self.palette["text"]),
            ))
            meta = f"{memory.get('source', '')} · {human_ts(memory.get('updated', ''))}"
            if memory.get("key"):
                meta += f" · {memory['key']}"
            panel.add_widget(_label(meta, self.palette, 7.5, "dim"))
            self.list_box.add_widget(panel)

    def _delete(self, memory_id: int) -> None:
        def confirmed(answer: bool) -> None:
            if not answer:
                return
            self.app.run_bg(lambda: self.core.memory.delete(memory_id), lambda _r: (self.app.toast("Deleted"), self.refresh()))
        self.app.confirm_async(T("Delete this memory?", "حذف هذه الذكرى؟"), str(memory_id), confirmed)

    def open_add(self) -> None:
        MemoryAddModal(self.app, self.refresh).open()

    def export_memory(self) -> None:
        def work() -> Any:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = EXPORT_DIR / f"memory_export_{stamp}.json"
            payload = self.core.memory.export_all()
            ok = atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), 0o600)
            return {"ok": ok, "path": str(path)}
        self.app.run_bg(work, lambda result: self.app.toast(
            (T("Exported to ", "تم التصدير إلى ") + str((result or {}).get("path", ""))) if (result or {}).get("ok")
            else T("Export failed", "فشل التصدير")
        ))

    def import_memory(self) -> None:
        self.app.run_bg(self.app.pick_path, self._import_path)

    def _import_path(self, path: Optional[str]) -> None:
        if not path:
            self.app.toast(T("No file selected.", "لم يتم اختيار ملف."))
            return

        def work() -> Any:
            raw = safe_json_load(Path(path), None)
            if not isinstance(raw, dict):
                return {"ok": False, "error": "Invalid memory export file"}
            count = self.core.memory.import_all(raw)
            return {"ok": True, "count": count}

        def done(result: Any) -> None:
            if (result or {}).get("ok"):
                self.app.toast(f"Imported {result.get('count', 0)} memories")
                self.refresh()
            else:
                self.app.toast(str((result or {}).get("error", "Import failed")))
        self.app.run_bg(work, done)


class MemoryAddModal(BoshraModal):
    def __init__(self, app: Any, on_change: Callable[[], Any]) -> None:
        super().__init__(app, T("Add memory", "إضافة ذكرى"), app.palette)
        self.on_change = on_change
        row = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(5))
        row.add_widget(_label(T("Category", "الفئة"), self.palette, 9, "dim"))
        self.category = Spinner(
            text="semantic", values=list(MEMORY_CATEGORIES), size_hint=(None, 1), width=dp(110),
            font_name=FONT, font_size=sp(9), background_normal="",
            background_color=_rgba(self.palette["accent"], 0.22), color=_rgba(self.palette["text"]),
        )
        row.add_widget(self.category)
        self.body.add_widget(row)
        self.content = _shaped_input(
            self.palette, multiline=True, font_size=sp(10.5),
            hint_text=T("What should Boshra remember?", "ماذا تتذكر بشرى؟"),
            size_hint_y=None, height=dp(110), padding=dp(7),
        )
        self.body.add_widget(self.content)
        self.key = _shaped_input(self.palette, hint_text=T("Optional key (profile fact)", "مفتاح اختياري"), size_hint_y=None, height=dp(36))
        self.body.add_widget(self.key)
        button = NeonButton(self.palette, text=T("Save locally", "احفظ محلياً"), size_hint_y=None, height=dp(38))
        button.bind(on_release=lambda *_: self._save())
        self.body.add_widget(button)

    def _save(self) -> None:
        content = (self.content.text or "").strip()
        if not content:
            return
        category = self.category.text
        key = (self.key.text or "").strip()

        def work() -> Any:
            return self.app.core.memory.add(category, content, key=key, source="user")

        def done(result: Any) -> None:
            if (result or {}).get("ok"):
                self.dismiss()
                self.on_change()
                self.app.toast(T("Saved locally", "تم الحفظ محلياً"))
            else:
                self.app.toast(str((result or {}).get("error", "Refused")))
        self.app.run_bg(work, done)


class VaultScreen(ScreenBase):
    key = "vault"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Knowledge Vault", "خزنة المعرفة"), [
            (T("Import", "استيراد"), self.import_document),
            (T("Refresh", "تحديث"), self.refresh),
        ])
        row = self.toolbar(38)
        self.query = _shaped_input(self.palette, hint_text=T("Search indexed passages", "ابحث في المقاطع المفهرسة"), size_hint=(1, 1))
        self.query.bind(on_text_validate=lambda *_: self.ask())
        row.add_widget(self.query)
        ask_btn = NeonButton(self.palette, text=T("Ask", "اسأل"), size_hint=(None, 1), width=dp(44), font_size=sp(9))
        ask_btn.bind(on_release=lambda *_: self.ask())
        row.add_widget(ask_btn)
        self.result_label = WrapLabel(
            text="", font_size=sp(8.5), color=_rgba(self.palette["dim"]),
            size_hint_y=None,
        )
        self.result_label.height = 0
        self.add_widget(self.result_label)
        self.list_scroll, self.list_box = self.list_area()
        self.refresh()

    def refresh(self) -> None:
        try:
            documents = self.core.vault.documents()
        except Exception as exc:  # noqa: BLE001
            LOG.error("vault list failed", exc)
            documents = []
        self.list_box.clear_widgets()
        self.list_box.add_widget(_label(f"{len(documents)} indexed documents", self.palette, 8.5, "dim"))
        if not documents:
            self.list_box.add_widget(WrapLabel(
                text=shape_text(T("Import a PDF, DOCX, text, CSV or image to build the local index.", "استورد PDF أو DOCX أو نصاً أو CSV أو صورة لبناء الفهرس المحلي.")),
                font_size=sp(10), color=_rgba(self.palette["dim"]),
            ))
            return
        for document in documents:
            panel = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None, alpha=0.50, radius=9,
                padding=[dp(7), dp(4), dp(5), dp(4)], spacing=dp(2), border_alpha=0.18,
            )
            panel.bind(minimum_height=panel.setter("height"))
            top = BoxLayout(size_hint_y=None, height=dp(22), spacing=dp(3))
            top.add_widget(_label(str(document["title"])[:70], self.palette, 10, "accent", bold=True))
            summary_btn = NeonButton(
                self.palette, text=T("Summary", "ملخص"), size_hint=(None, 1), width=dp(58), font_size=sp(7.5),
            )
            summary_btn.bind(on_release=lambda *_args, did=document["id"]: self.summarize(did))
            top.add_widget(summary_btn)
            delete_btn = NeonButton(
                self.palette, text=T("Delete", "حذف"), tone="danger", size_hint=(None, 1), width=dp(48), font_size=sp(7.5),
            )
            delete_btn.bind(on_release=lambda *_args, did=document["id"]: self.delete(did))
            top.add_widget(delete_btn)
            panel.add_widget(top)
            meta = (
                f"{document.get('mime', '?')} · {human_bytes(int(document.get('size') or 0))} · "
                f"{document.get('pages', 0)} pages · {human_ts(document.get('added', ''))}"
            )
            panel.add_widget(_label(meta, self.palette, 7.5, "dim"))
            self.list_box.add_widget(panel)

    def import_document(self) -> None:
        self.app.run_bg(self.app.pick_path, self._import_path)

    def _import_path(self, path: Optional[str]) -> None:
        if not path:
            self.app.toast(T("No file selected.", "لم يتم اختيار ملف."))
            return

        def work() -> Any:
            return self.app.core.tools.execute("vault_import", {"path": path})

        def done(result: Any) -> None:
            if result is None:
                return
            self.app.toast(result.render()[:200])
            self.refresh()
        self.app.run_bg(work, done)

    def ask(self) -> None:
        question = (self.query.text or "").strip()
        if not question:
            self.app.toast(T("Type a question first.", "اكتب سؤالاً أولاً."))
            return
        self.result_label.text = shape_text(T("Searching local passages...", "أبحث في المقاطع المحلية..."))

        def work() -> Any:
            return self.app.core.tools.execute("vault_answer", {"query": question})

        def done(result: Any) -> None:
            if result is None:
                return
            text = result.render()
            self.result_label.text = shape_text(text[:500])
            ResultModal(self.app, T("Vault answer", "إجابة الخزنة"), text).open()
        self.app.run_bg(work, done)

    def summarize(self, doc_id: int) -> None:
        def work() -> Any:
            return self.app.core.tools.execute("vault_summarize", {"document": str(doc_id)})

        def done(result: Any) -> None:
            if result is None:
                return
            ResultModal(self.app, T("Document summary", "ملخص المستند"), result.render()).open()
        self.app.run_bg(work, done)

    def delete(self, doc_id: int) -> None:
        def confirmed(answer: bool) -> None:
            if not answer:
                return

            def work() -> Any:
                ok = self.app.core.vault.delete_document(doc_id)
                return {"ok": ok, "message": "Document removed" if ok else "Delete failed"}
            self.app.run_bg(work, lambda _r: self.refresh())
        self.app.confirm_async(
            T("Delete this document from the vault?", "حذف هذا المستند من الخزنة؟"),
            T("The local index and copied file are removed. Original files are untouched.", "سيُحذف الفهرس المحلي والنسخة المستوردة. الملفات الأصلية لن تمس."),
            confirmed,
        )


class TextPromptModal(BoshraModal):
    def __init__(self, app: Any, title: str, hint: str, callback: Callable[[str], None],
                 initial: str = "") -> None:
        super().__init__(app, title, app.palette)
        self.callback = callback
        self.field = _shaped_input(app.palette, hint_text=hint, text=initial, size_hint_y=None, height=dp(38))
        self.body.add_widget(self.field)
        save = NeonButton(app.palette, text=T("Save", "حفظ"), size_hint_y=None, height=dp(36))
        save.bind(on_release=lambda *_: self._save())
        self.body.add_widget(save)

    def _save(self) -> None:
        value = (self.field.text or "").strip()
        self.dismiss()
        self.callback(value)


class ConversationBrowserModal(BoshraModal):
    """Browse conversation titles locally and safely resume a saved thread."""

    def __init__(self, app: Any) -> None:
        super().__init__(app, T("Conversations", "المحادثات"), app.palette)
        self.search = _shaped_input(
            app.palette, hint_text=T("Search conversation titles…", "ابحث في عناوين المحادثات…"),
            size_hint_y=None, height=dp(40),
        )
        self.body.add_widget(self.search)
        self.results = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(6))
        self.results.bind(minimum_height=self.results.setter("height"))
        self.body.add_widget(self.results)
        self._trigger = Clock.create_trigger(self._refresh, 0.2)
        self.search.bind(text=lambda *_: self._trigger())
        self._refresh()

    def _refresh(self, *_args: Any) -> None:
        self.results.clear_widgets()
        # Literal substring search: '%' and '_' are ordinary title characters.
        query = self.search.text.strip().lower()
        rows = self.app.core.store.query(
            "SELECT c.id, c.title, c.updated, COUNT(m.id) AS messages "
            "FROM conversations c LEFT JOIN messages m ON m.convo_id=c.id "
            "WHERE instr(lower(c.title), ?) > 0 GROUP BY c.id "
            "ORDER BY c.updated DESC, c.id DESC LIMIT 60", (query,),
        )
        if not rows:
            self.results.add_widget(_label(T("No conversations found.", "لا توجد محادثات مطابقة."), self.palette))
        for row in rows:
            active = int(row["id"]) == int(self.app.core.config.last_conversation_id or 0)
            title = ("● " if active else "") + str(row["title"])
            title += f"\n{human_ts(row['updated'])} · {row['messages']} " + T("messages", "رسائل")
            button = NeonButton(self.palette, text=shape_text(title), size_hint_y=None,
                                height=dp(58), font_size=sp(10))
            button.bind(on_release=lambda *_args, cid=int(row["id"]): self.app.ui_guard(lambda: self._resume(cid)))
            self.results.add_widget(button)

    def _resume(self, conversation_id: int) -> None:
        core = self.app.core
        if core.busy or core.orchestrator._processing:
            self.app.toast(T("Wait for the current reply to finish.", "انتظر اكتمال الرد الحالي."))
            return
        if not core.store.query_one("SELECT id FROM conversations WHERE id=?", (conversation_id,)):
            self._refresh()
            return
        core.ctx.take_pending()
        core.config.set("last_conversation_id", conversation_id)
        chat = self.app.ensure_screen("chat")
        chat.input.text = ""
        chat.load_history()
        self.app._ensure_welcome(chat)
        chat.restore_draft(focus=True)
        self.app.show_screen("chat")
        self.dismiss()


class ConversationSearchModal(BoshraModal):
    """Find a phrase in the visible conversation and jump to the message."""

    def __init__(self, app: Any, chat: Any) -> None:
        super().__init__(app, T("Find in conversation", "بحث في المحادثة"), app.palette,
                         height=min(Window.height * 0.78, dp(620)))
        self.chat = chat
        self.field = _shaped_input(
            app.palette, hint_text=T("Type words or a phrase…", "اكتب كلمة أو عبارة…"),
            size_hint_y=None, height=dp(40),
        )
        self.body.add_widget(self.field)
        self.count = _label("", app.palette, 8.5, "dim")
        self.body.add_widget(self.count)
        self.results = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(5))
        self.results.bind(minimum_height=self.results.setter("height"))
        self.body.add_widget(self.results)
        self._trigger = Clock.create_trigger(self._refresh, 0.18)
        self.field.bind(text=lambda *_: self._trigger())
        self._refresh()

    def _refresh(self, *_args: Any) -> None:
        self.results.clear_widgets()
        query = (self.field.text or "").strip()
        hits = self.chat.search_entries(query) if query else []
        self.count.text = shape_text(
            T(f"{len(hits)} matching messages", f"{len(hits)} رسالة مطابقة")
        )
        for entry in reversed(hits[-40:]):
            text = str(getattr(entry["body"], "_logical_text", "") or "").strip()
            needle = normalize_arabic(query.lower())
            position = normalize_arabic(text.lower()).find(needle)
            if position >= 0:
                start = max(0, position - 40)
                snippet = text[start:start + 150]
            else:
                snippet = text[:150]
            role = T("You", "أنت") if entry["role"] == "user" else "BOSHRA"
            button = NeonButton(
                self.palette, text=f"{role} · {human_ts(entry['ts'])}\n{snippet}",
                size_hint_y=None, height=dp(54), font_size=sp(8.6), alpha=0.14,
            )
            button.bind(
                on_release=lambda *_args, target=entry: self._jump(target)
            )
            self.results.add_widget(button)
        if not query:
            self.results.add_widget(_label(
                T("Start typing to search this conversation.", "ابدأ الكتابة للبحث في هذه المحادثة."),
                self.palette, 9, "dim",
            ))
        elif not hits:
            self.results.add_widget(_label(
                T("No message contains that phrase.", "لا توجد رسالة تحتوي هذه العبارة."),
                self.palette, 9, "dim",
            ))

    def _jump(self, entry: Dict[str, Any]) -> None:
        self.dismiss()
        self.chat.scroll_to_entry(entry)


class CommandListModal(BoshraModal):
    def __init__(self, app: Any, title: str, palette: Dict[str, str], core: Any) -> None:
        super().__init__(app, title, palette, height=min(Window.height * 0.92, dp(780)))
        self.text(T(
            "Every command runs through a registered tool with an explicit permission and risk level. "
            "Nothing is executed from model-generated code directly.",
            "كل أمر يُنفَّذ عبر أداة مسجّلة ذات إذن ومستوى خطورة محددين. لا يُنفَّذ كود مولّد من النموذج مباشرة.",
        ), 8.5, "dim")
        search = _shaped_input(palette, hint_text=T("Search tools and skills…", "ابحث عن أدوات ومهارات…"),
                               size_hint_y=None, height=dp(40))
        self.body.add_widget(search)
        self._catalog_rows = []
        self._search = search
        self._search_trigger = Clock.create_trigger(self._filter_catalog, 0.15)
        search.bind(text=lambda *_: self._search_trigger())
        tools = core.tools.specs()
        self.body.add_widget(_label(T(f"{len(tools)} registered tools", f"{len(tools)} أداة مسجّلة"),
                                    palette, 10.5, "accent", bold=True))
        for spec in tools:
            panel = GlassPanel(
                palette, orientation="vertical", size_hint_y=None, alpha=0.42, radius=8,
                padding=[dp(6), dp(3), dp(6), dp(4)], spacing=dp(1), border_alpha=0.12,
            )
            panel.bind(minimum_height=panel.setter("height"))
            panel.add_widget(_label(spec.name, palette, 9.2, "accent", bold=True))
            panel.add_widget(WrapLabel(
                text=shape_text(spec.description),
                font_size=sp(8.2), color=_rgba(palette["dim"]),
            ))
            flags = [f"risk:{spec.risk}"]
            if spec.needs_network:
                flags.append(T("network", "شبكة"))
            if spec.permissions:
                flags.append("perm:" + ", ".join(spec.permissions))
            panel.add_widget(_label(" · ".join(flags), palette, 7.2, "dim"))
            self.body.add_widget(panel)
            self._catalog_rows.append((panel, normalize_arabic(f"{spec.name} {spec.description} {spec.risk}".casefold())))
        skills = core.skills.list_skills()
        self.body.add_widget(_label(T(f"{len(skills)} skills", f"{len(skills)} مهارة"),
                                    palette, 10.5, "accent", bold=True))
        for skill in skills[:80]:
            panel = GlassPanel(
                palette, orientation="vertical", size_hint_y=None, alpha=0.38, radius=8,
                padding=[dp(6), dp(3), dp(6), dp(4)], spacing=dp(1), border_alpha=0.10,
            )
            panel.bind(minimum_height=panel.setter("height"))
            panel.add_widget(_label(
                f"{skill.get('name', '?')} · {skill.get('category', '')}", palette, 8.8, "text", bold=True,
            ))
            panel.add_widget(WrapLabel(
                text=shape_text(str(skill.get("description", ""))),
                font_size=sp(7.8), color=_rgba(palette["dim"]),
            ))
            command_names = [str(c.get("name", "")) for c in skill.get("commands", []) if c.get("name")]
            if command_names:
                panel.add_widget(_label(
                    "commands: " + ", ".join(command_names[:6]), palette, 7.0, "dim",
                ))
            self.body.add_widget(panel)
            self._catalog_rows.append((panel, normalize_arabic(str(skill).casefold())))

        self._catalog_order = list(reversed(self.body.children))[2:]
        self._match_count = _label("", palette, 8.5, "dim")
        self.body.add_widget(self._match_count, index=len(self.body.children) - 2)

    def _filter_catalog(self, *_args: Any) -> None:
        terms = normalize_arabic(self._search.text.casefold()).split()
        matching = {panel for panel, searchable in self._catalog_rows
                    if all(term in searchable for term in terms)}
        for widget in self._catalog_order:
            if widget.parent is self.body:
                self.body.remove_widget(widget)
        cards = {panel for panel, _ in self._catalog_rows}
        for widget in self._catalog_order:
            if widget in matching or (not terms and widget not in cards):
                self.body.add_widget(widget)
        self._match_count.text = shape_text(T(f"{len(matching)} results", f"{len(matching)} نتيجة"))


class AutomationScreen(ScreenBase):
    key = "flow"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Automation", "الأتمتة"), [
            (T("New", "جديد"), self.open_builder),
            (T("Save recording", "حفظ التسجيل"), self.save_recording),
            (T("Record", "تسجيل"), self.toggle_recording),
        ])
        controls = self.toolbar(34)
        for text, callback in [
            (T("Pause", "إيقاف مؤقت"), lambda: self.core.automation.pause()),
            (T("Resume", "استئناف"), lambda: self.core.automation.resume()),
            (T("Cancel run", "إلغاء"), lambda: self.core.automation.cancel()),
        ]:
            button = NeonButton(self.palette, text=text, size_hint_y=1, font_size=sp(8.5))
            button.bind(on_release=lambda *_args, cb=callback: self.app.ui_guard(cb))
            controls.add_widget(button)
        self.status = WrapLabel(
            text=shape_text(T("No routine running.", "لا توجد أتمتة قيد التنفيذ.")),
            font_size=sp(8.5), color=_rgba(self.palette["dim"]),
        )
        self.add_widget(self.status)
        self.list_scroll, self.list_box = self.list_area()
        self.refresh()

    def refresh(self) -> None:
        try:
            routines = self.core.automation.list_routines()
        except Exception as exc:  # noqa: BLE001
            LOG.error("routine list failed", exc)
            routines = []
        self.list_box.clear_widgets()
        rec = "recording ON" if self.core.config.recording else "recording off"
        self.list_box.add_widget(_label(
            f"{len(routines)} routines · {rec} · shadow previews on destructive steps",
            self.palette, 8.2, "dim",
        ))
        for routine in routines:
            panel = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None, alpha=0.50, radius=9,
                padding=[dp(7), dp(4), dp(5), dp(4)], spacing=dp(2), border_alpha=0.18,
            )
            panel.bind(minimum_height=panel.setter("height"))
            top = BoxLayout(size_hint_y=None, height=dp(22), spacing=dp(3))
            top.add_widget(_label(
                ("● " if routine["enabled"] else "○ ") + str(routine["name"])[:44],
                self.palette, 10, "accent" if routine["enabled"] else "dim", bold=True,
            ))
            run_btn = NeonButton(self.palette, text=T("Run", "تشغيل"), size_hint=(None, 1), width=dp(40), font_size=sp(8))
            run_btn.bind(on_release=lambda *_args, name=routine["name"]: self.run_routine(name))
            top.add_widget(run_btn)
            delete_btn = NeonButton(
                self.palette, text="✕", tone="danger", size_hint=(None, 1), width=dp(30), font_size=sp(9),
            )
            delete_btn.bind(on_release=lambda *_args, rid=routine["id"], name=routine["name"]: self.delete_routine(rid, name))
            top.add_widget(delete_btn)
            panel.add_widget(top)
            steps = routine.get("steps_parsed", [])
            summary = " → ".join(str(step.get("tool", "?")) for step in steps[:6])
            panel.add_widget(_label((summary or "no steps")[:120], self.palette, 7.8, "dim"))
            toggle_btn = Chip(
                self.palette,
                text=T("Disable", "تعطيل") if routine["enabled"] else T("Enable", "تمكين"),
                height=dp(22),
            )
            toggle_btn.bind(on_release=lambda *_args, rid=routine["id"], state=not routine["enabled"]: self.toggle_routine(rid, state))
            panel.add_widget(toggle_btn)
            self.list_box.add_widget(panel)

    def open_builder(self) -> None:
        RoutineBuilderModal(self.app, self.refresh).open()

    def run_routine(self, name: str) -> None:
        self.status.text = shape_text(T("Running: ", "تشغيل: ") + name)

        def work() -> Any:
            return self.core.automation.run_by_name(name)

        def done(result: Any) -> None:
            if not result:
                return
            self.status.text = shape_text(str(result.get("message", ""))[:400])
            ResultModal(self.app, T("Routine result", "نتيجة الأتمتة"), str(result.get("message", ""))).open()
            self.refresh()
        self.app.run_bg(work, done)

    def toggle_routine(self, routine_id: int, enabled: bool) -> None:
        self.core.automation.toggle_routine(routine_id, enabled)
        self.refresh()

    def delete_routine(self, routine_id: int, name: str) -> None:
        def confirmed(answer: bool) -> None:
            if answer:
                self.core.automation.delete_routine(routine_id)
                self.refresh()
        self.app.confirm_async(T("Delete routine?", "حذف الأتمتة؟"), name, confirmed)

    def toggle_recording(self) -> None:
        value = not bool(self.core.config.recording)
        self.core.config.set("recording", value)
        self.core.ctx.recording_steps.clear()
        self.app.toast(
            T("Recording enabled: tool calls will be captured.", "تم تفعيل التسجيل: ستُسجل نداءات الأدوات.")
            if value else T("Recording stopped.", "تم إيقاف التسجيل.")
        )
        self.refresh()

    def save_recording(self) -> None:
        steps = list(self.core.ctx.recording_steps)
        if not steps:
            self.app.toast(T("No recorded steps. Enable recording and run tools first.", "لا خطوات مسجلة. فعّل التسجيل ونفّذ أدوات أولاً."))
            return

        def save(name: str) -> None:
            if not name:
                return
            result = self.core.automation.save_recording(name)
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
            self.refresh()
        TextPromptModal(self.app, T("Save recording", "حفظ التسجيل"), T("Routine name", "اسم الأتمتة"), save).open()

    def on_workflow(self, payload: Dict[str, Any]) -> None:
        try:
            state = payload.get("state", "")
            name = payload.get("name", "")
            progress = int(float(payload.get("progress", 0.0)) * 100)
            step = payload.get("step", "")
            if state == "running":
                self.status.text = shape_text(f"{name} · {progress}% · {step}")
            else:
                self.status.text = shape_text(f"{name} · {payload.get('reason', 'done')}")
        except Exception:  # noqa: BLE001
            pass


class RoutineBuilderModal(BoshraModal):
    def __init__(self, app: Any, on_change: Callable[[], Any]) -> None:
        super().__init__(app, T("Build routine", "بناء أتمتة"), app.palette)
        self.on_change = on_change
        self.steps: List[Dict[str, Any]] = []
        self.name = _shaped_input(self.palette, hint_text=T("Routine name", "اسم الأتمتة"), size_hint_y=None, height=dp(36))
        self.body.add_widget(self.name)
        row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(4))
        tools = list(self.app.core.tools.names())
        self.tool = Spinner(
            text=tools[0] if tools else "", values=tools, size_hint=(1, 1),
            font_name=FONT, font_size=sp(9), background_normal="",
            background_color=_rgba(self.palette["accent"], 0.20), color=_rgba(self.palette["text"]),
        )
        row.add_widget(self.tool)
        self.body.add_widget(row)
        self.args = _shaped_input(
            self.palette, hint_text='{"minutes": 25}', size_hint_y=None, height=dp(36),
        )
        self.body.add_widget(self.args)
        add = NeonButton(self.palette, text=T("Add step", "إضافة خطوة"), size_hint_y=None, height=dp(33))
        add.bind(on_release=lambda *_: self._add_step())
        self.body.add_widget(add)
        self.steps_label = WrapLabel(text="", font_size=sp(8.5), color=_rgba(self.palette["dim"]))
        self.body.add_widget(self.steps_label)
        save = NeonButton(self.palette, text=T("Save routine", "حفظ الأتمتة"), size_hint_y=None, height=dp(38))
        save.bind(on_release=lambda *_: self._save())
        self.body.add_widget(save)

    def _add_step(self) -> None:
        raw = (self.args.text or "").strip() or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            self.app.toast(T("Invalid JSON: ", "JSON غير صالح: ") + str(exc))
            return
        self.steps.append({"tool": self.tool.text, "args": args})
        self.steps_label.text = shape_text("\n".join(
            f"{index + 1}. {step['tool']} {json.dumps(step['args'], ensure_ascii=False)}"
            for index, step in enumerate(self.steps)
        ))
        self.args.text = ""

    def _save(self) -> None:
        name = (self.name.text or "").strip()
        if not name or not self.steps:
            self.app.toast(T("Provide a name and at least one step.", "أدخل اسماً وخطوة واحدة على الأقل."))
            return
        result = self.app.core.automation.save_routine(name, self.steps)
        self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
        if (result or {}).get("ok"):
            self.dismiss()
            self.on_change()


class EmergencyScreen(ScreenBase):
    key = "sos"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Emergency & Rescue", "الطوارئ والإنقاذ"), [
            (T("Numbers", "الأرقام"), lambda: self.show("numbers")),
        ])
        grid = GridLayout(cols=2, size_hint_y=None, height=dp(148), spacing=dp(4))
        topics = list(self.core.emergency.TOPICS.items())
        for key, topic in topics[:8]:
            title = topic["title_ar"] if _UI_LANG == "ar" else (
                topic["title_en"] if _UI_LANG == "en" else f"{topic['title_en']} · {topic['title_ar']}"
            )
            button = NeonButton(self.palette, text=title, font_size=sp(8.5))
            button.bind(on_release=lambda *_args, query=key: self.show(query))
            grid.add_widget(button)
        self.add_widget(grid)
        self.scroll, self.box = self.list_area()
        self.show("")

    def show(self, query: str) -> None:
        def work() -> Any:
            return self.core.tools.execute("emergency", {"query": query})

        def done(result: Any) -> None:
            self.box.clear_widgets()
            if result is None:
                return
            text = result.render() if result.ok else (result.error or "Unavailable")
            self.box.add_widget(WrapLabel(
                text=shape_text(text), font_size=sp(10.5), color=_rgba(self.palette["text"]),
            ))
            if result.spoken and self.core.config.tts_enabled:
                self.core.tts.speak(result.spoken, detect_language(text))
        self.app.run_bg(work, done)


class PrivacyScreen(ScreenBase):
    key = "privacy"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Privacy Center", "مركز الخصوصية"), [
            (T("Encrypt", "تشفير"), self.encrypt_now),
            (T("Audit", "التدقيق"), self.export_audit),
        ])
        crypto = self.core.store.crypto
        self.crypto_label = _label(
            ("Encryption: " + crypto.status) if crypto.available else T("Encryption unavailable - local data is not encrypted at rest", "التشفير غير متاح - البيانات المحلية غير مشفرة"),
            self.palette, 8.2, "good" if crypto.available else "danger",
        )
        self.add_widget(self.crypto_label)
        self.add_widget(_label(T("Permissions", "الأذونات"), self.palette, 9.5, "accent", bold=True))
        perm_scroll = ScrollView(size_hint_y=0.55, do_scroll_x=False, bar_width=dp(2))
        perm_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(2))
        perm_box.bind(minimum_height=perm_box.setter("height"))
        perm_scroll.add_widget(perm_box)
        grants = self.core.permissions.grants()
        for permission, (risk, why, label) in PERMISSION_CATALOG.items():
            row = GlassPanel(
                self.palette, orientation="horizontal", size_hint_y=None, height=dp(38),
                alpha=0.42, radius=7, padding=[dp(5), dp(2)], spacing=dp(3), border_alpha=0.12,
            )
            info = BoxLayout(orientation="vertical", spacing=0)
            info.add_widget(_label(label, self.palette, 8.5, "text"))
            info.add_widget(_label(f"{permission} · {risk}", self.palette, 7, "dim"))
            row.add_widget(info)
            switch = ToggleSwitch(self.palette, active=bool(grants.get(permission)), width=dp(46))
            switch.bind(active=lambda _w, value, perm=permission, why=why, risk=risk: self._toggle(perm, bool(value), why, risk))
            row.add_widget(switch)
            perm_box.add_widget(row)
        self.add_widget(perm_scroll)
        head = BoxLayout(size_hint_y=None, height=dp(20))
        head.add_widget(_label(T("Audit log", "سجل التدقيق"), self.palette, 9.5, "accent", bold=True))
        self.add_widget(head)
        audit_scroll = ScrollView(size_hint_y=0.42, do_scroll_x=False, bar_width=dp(2))
        self.audit_box = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(1))
        self.audit_box.bind(minimum_height=self.audit_box.setter("height"))
        audit_scroll.add_widget(self.audit_box)
        self.add_widget(audit_scroll)
        self.refresh_audit()

    def _toggle(self, permission: str, granted: bool, why: str, risk: str) -> None:
        if not granted:
            self.app.run_bg(lambda: self.core.permissions.revoke(permission), lambda _r: self.app.toast(f"Revoked {permission}"))
            return

        def confirmed(answer: bool) -> None:
            if not answer:
                self.app.rebuild_ui()
                return
            self.app.run_bg(
                lambda: self.core.permissions.grant(permission, why),
                lambda _r: self.app.toast(f"Granted {permission}"),
            )
        self.app.confirm_async(
            f"{T('Grant permission?', 'منح الإذن؟')}\n{permission}",
            f"{why}\n{T('Risk', 'الخطورة')}: {risk.upper()}",
            confirmed,
        )

    def refresh_audit(self) -> None:
        self.audit_box.clear_widgets()
        rows = self.core.store.query("SELECT * FROM audit ORDER BY id DESC LIMIT 40")
        if not rows:
            self.audit_box.add_widget(_label(T("No audit entries yet.", "لا توجد مدخلات تدقيق بعد."), self.palette, 8.5, "dim"))
        for row in rows:
            line = f"{row['ts']} [{row['risk']}] {row['category']}:{row['action']} {str(row['detail'])[:80]}"
            self.audit_box.add_widget(WrapLabel(
                text=shape_text(line), font_size=sp(7.6), color=_rgba(self.palette["dim"]),
            ))

    def encrypt_now(self) -> None:
        self.app.run_bg(
            lambda: self.core.diagnostics.recover("encrypt"),
            lambda result: self.app.toast(str((result or {}).get("message", ""))),
        )

    def export_audit(self) -> None:
        def work() -> Any:
            rows = self.core.store.query("SELECT * FROM audit ORDER BY id DESC LIMIT 5000")
            path = EXPORT_DIR / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ok = atomic_write_text(path, json.dumps(rows, ensure_ascii=False, indent=2), 0o600)
            return {"ok": ok, "path": str(path)}
        self.app.run_bg(work, lambda result: self.app.toast(str((result or {}).get("path") or "Export failed")))


class DiagnosticsScreen(ScreenBase):
    key = "diag"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Diagnostics", "التشخيص"), [
            (T("Run", "تشغيل"), self.run_checks),
            (T("Report", "تقرير"), self.export_report),
        ])
        recover = BoxLayout(size_hint_y=None, height=dp(30), spacing=dp(3))
        for label_text, action in [
            (T("Vacuum", "ضغط"), "vacuum"),
            (T("Reindex", "فهرسة"), "reindex"),
            (T("Disable bad", "تعطيل المعطوب"), "disable-unhealthy"),
            (T("Cache", "ذاكرة مؤقتة"), "clear-cache"),
        ]:
            button = NeonButton(self.palette, text=label_text, font_size=sp(7.8))
            button.bind(on_release=lambda *_args, act=action: self.recover(act))
            recover.add_widget(button)
        self.add_widget(recover)
        self.scroll, self.box = self.list_area()
        self.run_checks()

    def run_checks(self) -> None:
        self.box.clear_widgets()
        self.box.add_widget(_label(T("Running checks...", "جارٍ الفحص..."), self.palette, 9, "dim"))

        def work() -> Any:
            return self.core.diagnostics.run_all()

        def done(checks: Any) -> None:
            self.box.clear_widgets()
            if not checks:
                self.box.add_widget(_label(T("Diagnostics unavailable.", "التشخيص غير متاح."), self.palette, 9, "danger"))
                return
            for check in checks:
                status = check.get("status", "warn")
                color = "good" if status == "ok" else ("danger" if status == "fail" else "accent2")
                panel = GlassPanel(
                    self.palette, orientation="vertical", size_hint_y=None, alpha=0.42, radius=8,
                    padding=[dp(6), dp(3)], spacing=0, border_alpha=0.14,
                )
                panel.bind(minimum_height=panel.setter("height"))
                panel.add_widget(_label(f"{status.upper()} · {check.get('name', '')}", self.palette, 8.8, color, bold=True))
                panel.add_widget(WrapLabel(
                    text=shape_text(str(check.get("detail", ""))[:300]),
                    font_size=sp(8.2), color=_rgba(self.palette["dim"]),
                ))
                self.box.add_widget(panel)
        self.app.run_bg(work, done)

    def recover(self, action: str) -> None:
        self.app.run_bg(
            lambda: self.core.diagnostics.recover(action),
            lambda result: (self.app.toast(str((result or {}).get("message", ""))), self.run_checks()),
        )

    def export_report(self) -> None:
        self.app.run_bg(
            lambda: self.core.diagnostics.export_report(),
            lambda result: self.app.toast(str((result or {}).get("path") or (result or {}).get("error") or "")),
        )


def _safe_extract_zip(archive: Path, destination: Path) -> List[str]:
    extracted: List[str] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if not is_within(target, destination):
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(target))
    return extracted


def _safe_extract_tar(archive: Path, destination: Path) -> List[str]:
    extracted: List[str] = []
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            target = (destination / member.name).resolve()
            if not is_within(target, destination):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as dst:
                shutil.copyfileobj(source, dst)
            extracted.append(str(target))
    return extracted


def download_model_asset(url: str, category: str,
                         progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """Download a model/voice archive from a direct URL into its asset folder."""
    import requests

    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return {"ok": False, "error": "Enter a direct http(s) download URL."}
    if category not in ASSET_NAMES:
        return {"ok": False, "error": "Unknown model asset category."}
    folders = ensure_asset_dirs()
    destination = folders.get(category)
    if destination is None:
        return {"ok": False, "error": "No writable model asset folder is available."}
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"ok": False, "error": f"Cannot create {destination}: {exc}"}
    name = Path(urlparse(url).path).name or f"{category}.bin"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]
    target = destination / name
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(20, 120),
                          headers={"User-Agent": "BoshraAI/4.0"}) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=512 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    received += len(chunk)
                    if progress:
                        with contextlib.suppress(Exception):
                            progress(received, total)
        partial.replace(target)
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(OSError):
            partial.unlink()
        return {"ok": False, "error": f"Download failed: {exc}"}

    lower = name.lower()
    extracted: List[str] = []
    try:
        if lower.endswith(".zip"):
            extracted = _safe_extract_zip(target, destination)
        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
            extracted = _safe_extract_tar(target, destination)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Archive could not be extracted: {exc}"}

    if lower.endswith(".gguf"):
        try:
            with target.open("rb") as handle:
                magic = handle.read(4)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        if magic != b"GGUF":
            with contextlib.suppress(OSError):
                target.unlink()
            return {"ok": False, "error": "The downloaded file is not a valid GGUF model."}

    verified_gguf = [path for path in extracted if path.lower().endswith(".gguf")]
    voices = [path for path in extracted if path.lower().endswith(".onnx")]
    return {
        "ok": True,
        "path": str(target),
        "extracted": len(extracted),
        "gguf": verified_gguf[:4],
        "voices": voices[:8],
        "message": f"Downloaded {name} to {destination}"
                   + (f" ({len(extracted)} files extracted)" if extracted else ""),
    }


def build_asset_status(_core: Any = None, report: Optional[AssetReport] = None) -> List[Dict[str, Any]]:
    """Localized UI rows backed by :func:`inspect_assets` (never raises)."""
    labels = {
        "phi3": T("Phi-3 GGUF", "Phi-3 GGUF"),
        "vosk-model-ar": T("Vosk Arabic", "Vosk العربية"),
        "vosk-model-en": T("Vosk English", "Vosk الإنجليزية"),
        "piper-tts": T("Piper TTS", "Piper TTS"),
        "fonts": T("Fonts", "الخطوط"),
    }
    if report is None:
        try:
            report = inspect_assets()
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"asset report failed: {exc}")
            return []
    rows: List[Dict[str, Any]] = []
    for status in report.statuses:
        row = status.as_dict()
        row["label"] = labels.get(status.key, status.label)
        if not status.ok and status.missing:
            missing = ", ".join(status.missing)
            row["detail"] = T(f"missing: {missing}", f"مفقود: {missing}")
        rows.append(row)
    return rows


class ModelsScreen(ScreenBase):
    key = "models"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Local Models", "النماذج المحلية"), [
            (T("Scan", "فحص"), self.scan),
            (T("Import", "استيراد"), self.import_model),
            (T("Folder", "المجلد"), self.show_folder),
            (T("Unload", "تفريغ"), self.unload),
        ])
        self.engine_label = WrapLabel(text="", font_size=sp(8.5), color=_rgba(self.palette["accent"]))
        self.add_widget(self.engine_label)
        profile_row = self.toolbar(30)
        for name in VALID_PROFILES:
            chip = Chip(self.palette, text=name, height=dp(26))
            chip.bind(on_release=lambda *_args, value=name: self.set_profile(value))
            profile_row.add_widget(chip)
        self.list_scroll, self.list_box = self.list_area()
        self.voices_header: Optional[Any] = None
        self.refresh()

    def set_profile(self, profile: str) -> None:
        self.core.config.set("profile", profile)
        self.app.toast(T("Profile updated. Reload the model to apply context/thread changes.", "تم تحديث الملف. أعد تحميل النموذج لتطبيق التغييرات."))
        self.refresh()

    def scan(self) -> None:
        def work() -> Any:
            return self.core.models.scan(force=True)

        def done(_result: Any) -> None:
            self.refresh()
            self.app.toast(T("Model scan complete", "تم فحص النماذج"))
        self.app.run_bg(work, done)

    def import_model(self) -> None:
        self.app.run_bg(self.app.pick_path, self._import_path)

    def _import_path(self, path: Optional[str]) -> None:
        if not path:
            self.app.toast(T("No model selected.", "لم يتم اختيار نموذج."))
            return

        def work() -> Any:
            return self.core.models.import_model(path)

        def done(result: Any) -> None:
            self.app.toast(str((result or {}).get("message") or (result or {}).get("error") or ""))
            self.refresh()
        self.app.run_bg(work, done)

    def refresh(self) -> None:
        models = self.core.models.scan()
        folders = ensure_asset_dirs()
        root = folders.get("phi3").parent if folders.get("phi3") else ASSETS_DIR
        if not self.core.config.assets_path:
            self.core.config.set("assets_path", str(root))
        self.engine_label.text = shape_text(
            self.core.llm.status + f"\nprofile: {self.core.config.profile} · "
            f"ctx={self.core.resources.profile['n_ctx']} · threads={self.core.resources.profile['n_threads']}\n"
            + T("assets folder: ", "مجلد الأصول: ") + str(root)
        )
        self.list_box.clear_widgets()
        self.list_box.add_widget(_label(
            T("MODEL ASSETS", "أصول النماذج"), self.palette, 9.5, "accent", bold=True,
        ))
        for entry in build_asset_status(self.core):
            panel = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None, alpha=0.44, radius=8,
                padding=[dp(6), dp(3), dp(6), dp(4)], spacing=dp(1), border_alpha=0.14,
            )
            panel.bind(minimum_height=panel.setter("height"))
            head = BoxLayout(size_hint_y=None, height=dp(22), spacing=dp(3))
            head.add_widget(_label(
                ("● " if entry["ok"] else "○ ") + entry["label"],
                self.palette, 9.0, "good" if entry["ok"] else "danger", bold=True,
            ))
            if entry["key"] in ASSET_NAMES:
                download = NeonButton(
                    self.palette, text=T("Download", "تنزيل"), size_hint=(None, 1),
                    width=dp(56), font_size=sp(7.2),
                )
                download.bind(
                    on_release=lambda *_args, key=entry["key"], name=entry["label"]:
                    self.app.ui_guard(lambda: self.ask_download(key, name))
                )
                head.add_widget(download)
            panel.add_widget(head)
            panel.add_widget(_label(str(entry["detail"]), self.palette, 7.8, "dim"))
            panel.add_widget(_label(str(entry["path"]), self.palette, 7.0, "dim", shorten=True))
            self.list_box.add_widget(panel)

        self.list_box.add_widget(_label(
            T("GGUF MODELS", "نماذج GGUF"), self.palette, 9.5, "accent", bold=True,
        ))
        if not models:
            self.list_box.add_widget(WrapLabel(
                text=shape_text(T(
                    "No GGUF model found. Import a Phi-3 / Gemma quantized GGUF, download one into the phi3 folder, or place it in BoshraAI/models. Boshra stays fully functional with the deterministic heuristic engine.",
                    "لم يُعثر على نموذج GGUF. استورد نموذج Phi-3 أو Gemma مكمماً، أو نزّل واحداً إلى مجلد phi3. تبقى بشرى عاملة بمحرك احتياطي حتمي.",
                )),
                font_size=sp(9.0), color=_rgba(self.palette["dim"]),
            ))
        selected = self.core.models.selected()
        for model in models:
            panel = GlassPanel(
                self.palette, orientation="vertical", size_hint_y=None, alpha=0.50, radius=9,
                padding=[dp(7), dp(4), dp(5), dp(4)], spacing=dp(2), border_alpha=0.18,
            )
            panel.bind(minimum_height=panel.setter("height"))
            top = BoxLayout(size_hint_y=None, height=dp(22), spacing=dp(3))
            marker = "●" if selected and selected.path == model.path else "○"
            top.add_widget(_label(f"{marker} {model.name}", self.palette, 9.5, "accent", bold=True))
            select_btn = NeonButton(self.palette, text=T("Select", "اختيار"), size_hint=(None, 1), width=dp(46), font_size=sp(7.5))
            select_btn.bind(on_release=lambda *_args, path=model.path: self.select(path))
            top.add_widget(select_btn)
            verify_btn = NeonButton(self.palette, text=T("Verify", "تحقق"), size_hint=(None, 1), width=dp(46), font_size=sp(7.5))
            verify_btn.bind(on_release=lambda *_args, path=model.path: self.verify(path))
            top.add_widget(verify_btn)
            panel.add_widget(top)
            panel.add_widget(_label(
                f"{human_bytes(model.size)} · {model.quant} · GGUF {'ok' if model.magic_ok else 'invalid'}",
                self.palette, 7.6, "dim",
            ))
            self.list_box.add_widget(panel)

        voices: List[Dict[str, Any]] = []
        with contextlib.suppress(Exception):
            voices = self.core.tts.piperian.list_voices()
        self.voices_header = _label(
            T("PIPER VOICES", "أصوات Piper"), self.palette, 9.5, "accent", bold=True,
        )
        self.list_box.add_widget(self.voices_header)
        if not voices:
            self.list_box.add_widget(_label(
                T("No .onnx voice found in piper-tts.", "لا يوجد صوت .onnx في مجلد piper-tts."),
                self.palette, 7.8, "dim",
            ))
        selected_voice = str(getattr(self.core.config, "piper_voice", "") or "")
        for voice in voices:
            chosen = bool(selected_voice) and os.path.realpath(selected_voice) == os.path.realpath(str(voice["path"]))
            panel = GlassPanel(
                self.palette, orientation="horizontal", size_hint_y=None, height=dp(30),
                alpha=0.40, radius=8, padding=[dp(6), dp(2)], spacing=dp(3), border_alpha=0.12,
            )
            panel.add_widget(_label(
                ("● " if chosen else "○ ") + f"{voice['name']} · {voice['lang'].upper()} · {human_bytes(int(voice['size']))}",
                self.palette, 7.6, "accent" if chosen else "text", shorten=True,
            ))
            use_btn = NeonButton(
                self.palette, text=T("In use", "مستخدم") if chosen else T("Use", "استخدام"),
                size_hint=(None, 1), width=dp(50), font_size=sp(7.2),
            )
            use_btn.disabled = chosen
            use_btn.bind(
                on_release=lambda *_args, item=voice: self.app.ui_guard(
                    lambda: self.use_voice(item)
                )
            )
            panel.add_widget(use_btn)
            test_btn = NeonButton(
                self.palette, text=T("Test", "اختبار"), size_hint=(None, 1),
                width=dp(46), font_size=sp(7.2),
            )
            test_btn.bind(
                on_release=lambda *_args, item=voice: self.app.ui_guard(
                    lambda: self.test_voice(item)
                )
            )
            panel.add_widget(test_btn)
            self.list_box.add_widget(panel)

        load = NeonButton(self.palette, text=T("Load selected model now", "تحميل النموذج المحدد الآن"), size_hint_y=None, height=dp(36))
        load.bind(on_release=lambda *_: self.load_model())
        self.list_box.add_widget(load)

    def select(self, path: str) -> None:
        self.core.models.select(path)
        self.refresh()
        self.app.toast(T("Model selected", "تم اختيار النموذج"))

    def verify(self, path: str) -> None:
        self.app.run_bg(
            lambda: self.core.models.verify(path),
            lambda result: ResultModal(self.app, T("Model verification", "التحقق من النموذج"), json.dumps(result or {}, ensure_ascii=False, indent=2)).open(),
        )

    def load_model(self) -> None:
        modal = BoshraModal(self.app, T("Loading model", "تحميل النموذج"), self.palette,
                            height=min(Window.height * 0.46, dp(320)))
        label = modal.text(T(
            "Loading the selected GGUF into memory. This can take a minute on 8 GB devices.",
            "جارٍ تحميل نموذج GGUF المحدد إلى الذاكرة. قد يستغرق دقيقة على أجهزة 8 غيغابايت.",
        ), 9.5)
        modal.open()

        def work() -> Any:
            return self.core.llm.load_selected()

        def done(result: Any) -> None:
            with contextlib.suppress(Exception):
                modal.dismiss()
            message = str((result or {}).get("message") or (result or {}).get("error") or "")
            if label is not None:
                label.text = shape_text(message)
            self.app.toast(message)
            self.refresh()

        self.app.run_bg(work, done)

    def unload(self) -> None:
        self.core.llm.unload()
        self.refresh()
        self.app.toast(T("Model unloaded; heuristic fallback active.", "تم تفريغ النموذج؛ المحرك الاحتياطي نشط."))

    def show_folder(self) -> None:
        folders = ensure_asset_dirs()
        modal = BoshraModal(self.app, T("Model folders", "مجلدات النماذج"), self.palette,
                            height=min(Window.height * 0.66, dp(520)))
        modal.text(T(
            "Place each model in its folder. The app scans these folders on every start.",
            "ضع كل نموذج في مجلده. يفحص التطبيق هذه المجلدات عند كل تشغيل.",
        ), 8.8, "dim")
        for key in ASSET_NAMES:
            modal.text(f"{key}:\n{folders.get(key, ASSETS_DIR / key)}", 8.2)
        modal.button(T("Recreate folders", "إعادة إنشاء المجلدات"), self._recreate_folders)
        modal.button(T("Set custom assets folder", "تعيين مجلد أصول مخصص"),
                     lambda: self.ask_folder())
        modal.open()

    def _recreate_folders(self) -> None:
        ensure_asset_dirs()
        self.app.toast(T("Model folders verified", "تم التحقق من مجلدات النماذج"))
        self.refresh()

    def ask_folder(self) -> None:
        def apply(value: str) -> None:
            value = (value or "").strip()
            if not value:
                return
            root = Path(value).expanduser()
            try:
                root.mkdir(parents=True, exist_ok=True)
                for name in ASSET_NAMES:
                    (root / name).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.app.toast(T(f"Cannot use that folder: {exc}", f"تعذر استخدام هذا المجلد: {exc}"), 5)
                return
            self.core.config.set("assets_path", str(root))
            self.app.toast(T(f"Assets folder saved: {root}", f"حُفظ مجلد الأصول: {root}"), 5)
            self.refresh()

        TextPromptModal(
            self.app, T("Assets folder", "مجلد الأصول"),
            T("/sdcard/assets or any writable folder", "/sdcard/assets أو أي مجلد قابل للكتابة"),
            apply, initial=str(self.core.config.assets_path or ""),
        ).open()

    def ask_download(self, category: str, label: str) -> None:
        def start(url: str) -> None:
            if url:
                self._download(category, label, url)

        TextPromptModal(
            self.app, T("Download ", "تنزيل ") + label,
            T("Direct https URL to a .gguf, .onnx, .zip or .tar.gz file",
              "رابط https مباشر لملف .gguf أو .onnx أو .zip أو .tar.gz"),
            start,
        ).open()

    def _download(self, category: str, label: str, url: str) -> None:
        modal = BoshraModal(self.app, T("Downloading ", "جارٍ تنزيل ") + label, self.palette,
                            height=min(Window.height * 0.44, dp(300)))
        status = modal.text(T("Starting download...", "بدء التنزيل..."), 9.2)
        modal.text(T(
            "Downloads go directly into the model folder and archives are extracted safely. "
            "Large models can take a long time on mobile data.",
            "تُنزَّل الملفات مباشرة إلى مجلد النموذج وتُستخرج الأرشيفات بأمان. قد تستغرق النماذج الكبيرة وقتاً طويلاً.",
        ), 8.0, "dim")
        modal.open()

        def progress(done: int, total: int) -> None:
            if total > 0:
                message = f"{human_bytes(done)} / {human_bytes(total)} · {done * 100 // total}%"
            else:
                message = human_bytes(done)
            self.app._ui_after(self._set_label, status, message)

        def work() -> Any:
            return download_model_asset(url, category, progress)

        def done(result: Any) -> None:
            result = result or {}
            with contextlib.suppress(Exception):
                modal.dismiss()
            message = str(result.get("message") or result.get("error") or "")
            self.app.toast(message, 6)
            self.refresh()

        self.app.run_bg(work, done)

    @staticmethod
    def _set_label(label: WrapLabel, text: str) -> None:
        with contextlib.suppress(Exception):
            label.text = shape_text(text)

    def scroll_to_voices(self) -> None:
        def scroll(_dt: float) -> None:
            if self.voices_header is not None:
                with contextlib.suppress(Exception):
                    self.list_scroll.scroll_to(self.voices_header, padding=dp(18), animate=False)

        Clock.schedule_once(scroll, 0.12)

    def use_voice(self, voice: Dict[str, Any]) -> None:
        path = str(voice.get("path", ""))
        lang = str(voice.get("lang") or "").lower()
        piper = self.core.tts.piperian
        if not piper.set_selected(path, lang):
            self.app.toast(T("That voice file is no longer available.", "ملف الصوت هذا لم يعد متاحاً."), 5)
            self.refresh()
            return
        config = self.core.config
        config.set("piper_voice", piper.selected_path)
        config.set("piper_voice_lang", piper.selected_lang)
        name = str(voice.get("name") or Path(path).stem)
        if not piper.available:
            self.refresh()
            self.app.toast(T(
                f"Saved as the assistant voice: {name}. Piper needs its binary and an audio player to speak. {piper.status}",
                f"حُفظ كصوت المساعد: {name}. يحتاج Piper إلى ملفه التنفيذي ومشغّل صوت للنطق. {piper.status}",
            ), 7)
            return
        config.set("tts_voice", "piper")

        def done(_result: Any) -> None:
            self.refresh()
            if config.tts_enabled:
                self.app.toast(T(f"Assistant voice: {name}", f"صوت المساعد: {name}"), 5)
            else:
                self.app.toast(T(
                    "Assistant voice saved. Speech output is off; enable it in Settings to hear it.",
                    "حُفظ صوت المساعد. الإخراج الصوتي متوقف؛ فعّله من الإعدادات لسماعه.",
                ), 6)

        self.app.run_bg(self.core.reconfigure_tts, done)

    def test_voice(self, voice: Dict[str, Any]) -> None:
        sample = (
            "مرحباً، هذه تجربة صوت بشرى."
            if voice.get("lang") == "ar"
            else "Hello, this is a Boshra voice test."
        )
        piper = self.core.tts.piperian

        def work() -> Any:
            if piper.binary:
                wav = piper.synthesize(sample, str(voice.get("path", "")), float(self.core.config.tts_rate))
                if wav is not None:
                    return {"ok": True, "wav": str(wav)}
            return {"ok": False}

        def done(result: Any) -> None:
            result = result or {}
            if result.get("ok"):
                try:
                    from kivy.core.audio import SoundLoader
                    sound = SoundLoader.load(str(result.get("wav")))
                    if sound is not None:
                        sound.play()
                        self.app.toast(T("Playing voice test", "جارٍ تشغيل اختبار الصوت"))
                        return
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(f"voice test playback failed: {exc}")
                self.app.toast(T("Voice rendered, but playback is unavailable.",
                                 "تم توليد الصوت، لكن التشغيل غير متاح."), 5)
                return
            if self.core.tts.available:
                self.core.tts.speak(sample, str(voice.get("lang") or "en"))
                self.app.toast(T(
                    "Piper binary not found; testing with the system voice instead.",
                    "لم يُعثر على ملف Piper التنفيذي؛ يتم الاختبار بصوت النظام بدلاً منه.",
                ), 5)
            else:
                self.app.toast(T(
                    "Cannot test: add the piper binary, or enable a system TTS engine.",
                    "تعذر الاختبار: أضف ملف Piper التنفيذي أو فعّل محرك نطق النظام.",
                ), 6)

        self.app.run_bg(work, done)


class SettingsScreen(ScreenBase):
    key = "settings"

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self.build_header(T("Settings", "الإعدادات"), [
            (T("About", "حول"), self.about),
        ])
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False, bar_width=dp(2))
        self.box = BoxLayout(
            orientation="vertical", size_hint_y=None, spacing=dp(5),
            padding=[dp(1), dp(1), dp(1), dp(16)],
        )
        self.box.bind(minimum_height=self.box.setter("height"))
        scroll.add_widget(self.box)
        self.add_widget(scroll)
        self.build()

    def _section(self, title: str) -> None:
        self.box.add_widget(_label(title, self.palette, 9.5, "accent", bold=True, size_hint_y=None, height=dp(20)))

    def _row(self, widget: Any, height: int = 38, title: str = "") -> GlassPanel:
        panel = GlassPanel(
            self.palette, orientation="horizontal", size_hint_y=None, height=dp(height),
            alpha=0.42, radius=8, padding=[dp(6), dp(2)], spacing=dp(4), border_alpha=0.14,
        )
        if title:
            panel.add_widget(_label(title, self.palette, 8.4, "text", shorten=True))
        panel.add_widget(widget)
        self.box.add_widget(panel)
        return panel

    def _chip_row(self, title: str, options: Sequence[Tuple[str, str]], current: str,
                  callback: Callable[[str], Any]) -> None:
        self._section(title)
        grid = GridLayout(cols=3, size_hint_y=None, spacing=dp(3))
        grid.bind(minimum_height=grid.setter("height"))
        chips: List[Tuple[str, Chip]] = []

        def select(value: str) -> None:
            for option_value, option_chip in chips:
                option_chip.background_color = _rgba(
                    self.palette["accent"], 0.34 if option_value == value else 0.13
                )
            callback(value)

        for value, label in options:
            chip = Chip(self.palette, text=label, height=dp(28))
            chip.size_hint_x = 1
            chip.background_color = _rgba(self.palette["accent"], 0.34 if value == current else 0.13)
            chip.bind(on_release=lambda *_args, val=value: select(val))
            chips.append((value, chip))
            grid.add_widget(chip)
        self.box.add_widget(grid)

    def _switch_row(self, title: str, active: bool, callback: Callable[[bool], Any]) -> None:
        switch = ToggleSwitch(self.palette, active=bool(active), width=dp(46))
        switch.bind(active=lambda _w, value: callback(bool(value)))
        self._row(switch, 38, title)

    def _slider_row(self, title: str, value: float, minimum: float, maximum: float,
                    callback: Callable[[float], Any], step: float = 0.1) -> None:
        panel = GlassPanel(
            self.palette, orientation="vertical", size_hint_y=None, height=dp(50),
            alpha=0.42, radius=8, padding=[dp(6), dp(2)], spacing=0, border_alpha=0.14,
        )
        panel.add_widget(_label(title, self.palette, 8.2, "dim", shorten=True, height=dp(14)))
        slider = Slider(min=minimum, max=maximum, value=value, step=step,
                        size_hint_y=None, height=dp(30))
        slider.bind(on_touch_up=lambda _w, _t: callback(float(slider.value)))
        panel.add_widget(slider)
        self.box.add_widget(panel)

    def build(self) -> None:
        cfg = self.core.config
        self._chip_row(T("Theme", "المظهر"), [(name, name.title()) for name in PALETTES], cfg.theme, self.app.apply_theme)
        self._chip_row(
            T("Interface language", "لغة الواجهة"),
            [("bilingual", "Bilingual"), ("en", "English"), ("ar", "العربية")],
            cfg.ui_lang, self.set_ui_lang,
        )
        self._switch_row(T("Animations", "الحركة"), cfg.animations, self.set_animations)
        self._slider_row(T("Shader intensity", "شدة التظليل"), cfg.shader_intensity, 0.2, 1.5, self.set_intensity)
        self._chip_row(
            T("Button shape", "شكل الأزرار"),
            [("glass", "Glass"), ("rounded", "Rounded"), ("pill", "Pill")],
            cfg.button_style, self.set_button_style,
        )
        self._section(T("Living sphere", "الكرة الحية"))
        self._chip_row(
            T("Sphere style", "نمط الكرة"),
            [("liquid", "Liquid"), ("particles", "Particles"), ("plasma", "Plasma")],
            cfg.sphere_style, self.set_sphere_style,
        )
        self._chip_row(
            T("Sphere colour", "لون الكرة"),
            [("cyan", "Cyan"), ("magenta", "Magenta"), ("gold", "Gold"),
             ("green", "Green"), ("violet", "Violet")],
            cfg.sphere_color, self.set_sphere_color,
        )

        self._section(T("Performance", "الأداء"))
        self._chip_row(
            T("Hardware profile", "ملف الموارد"),
            [(name, name) for name in VALID_PROFILES], cfg.profile, self.set_profile,
        )
        self._switch_row(T("Auto-unload model under memory/thermal pressure", "تفريغ النموذج عند الضغط"), cfg.auto_unload_model, self.set_auto_unload)

        self._section(T("Intelligence", "الذكاء"))
        self._chip_row(
            T("Online mode", "وضع الشبكة"),
            [("local", "Local only"), ("online", "Online"), ("auto", "Automatic")],
            cfg.online_mode, self.set_online_mode,
        )
        self._chip_row(
            T("Conversation language", "لغة المحادثة"),
            [("auto", "Auto"), ("en", "English"), ("ar", "العربية")],
            cfg.language, self.set_language,
        )
        model = self.core.models.selected()
        self._row(
            WrapLabel(
                text=shape_text((model.name if model else T("No GGUF model", "لا يوجد نموذج GGUF")) + " · " + self.core.llm.engine_name),
                font_size=sp(8), color=_rgba(self.palette["dim"]),
            ),
            46, T("Model", "النموذج"),
        )
        self._asset_section()

        self._section(T("Sound & feedback", "الصوت والتنبيهات"))
        self._switch_row(T("Menu & button sounds", "أصوات القوائم والأزرار"), cfg.ui_sound, self.set_ui_sound)
        self._slider_row(T("Click volume", "مستوى صوت النقر"), cfg.ui_sound_volume, 0.0, 1.0, self.set_ui_volume, step=0.05)
        test_sound = NeonButton(self.palette, text=T("Test sound", "تجربة الصوت"), size_hint_y=None, height=dp(32))
        test_sound.sound_kind = "silent"
        test_sound.bind(on_release=lambda *_: self.test_sound())
        self.box.add_widget(test_sound)
        self.box.add_widget(WrapLabel(text=shape_text(T(
            "Tones are generated on this device; no audio files are downloaded.",
            "تُولَّد النغمات على هذا الجهاز، ولا تُنزَّل أي ملفات صوتية.",
        )), font_size=sp(7.6), color=_rgba(self.palette["dim"])))

        self._section(T("Voice", "الصوت"))
        self._switch_row(T("Speech output (TTS)", "الإخراج الصوتي"), cfg.tts_enabled, self.set_tts)
        self._slider_row(T("Speech rate", "سرعة النطق"), cfg.tts_rate, 0.6, 1.8, self.set_tts_rate)
        voice = Spinner(
            text=cfg.tts_voice, values=["auto", "piper", "android", "desktop"],
            size_hint=(1, 1), font_name=FONT, font_size=sp(8.5), background_normal="",
            background_color=_rgba(self.palette["accent"], 0.2), color=_rgba(self.palette["text"]),
        )
        voice.bind(text=lambda _w, value: self.set_tts_voice(value))
        self._row(voice, 38, T("TTS backend", "محرك النطق"))
        choose_voice = NeonButton(
            self.palette, text=T("Choose assistant voice", "اختيار صوت المساعد"),
            size_hint_y=None, height=dp(32),
        )
        choose_voice.bind(on_release=lambda *_: self.choose_voice())
        self.box.add_widget(choose_voice)
        wake = _shaped_input(self.palette, text=cfg.wake_word, size_hint=(1, 1))
        save_wake = NeonButton(self.palette, text=T("Set", "ضبط"), size_hint=(None, 1), width=dp(40), font_size=sp(8))
        save_wake.bind(on_release=lambda *_: self.set_wake(wake.text))
        wrap = BoxLayout(spacing=dp(3))
        wrap.add_widget(wake)
        wrap.add_widget(save_wake)
        self._row(wrap, 38, T("Wake word", "كلمة التنبيه"))
        self._switch_row(T("Continuous conversation", "محادثة مستمرة"), cfg.continuous_conversation, self.set_continuous)
        self.voice_status = WrapLabel(
            text=shape_text("Voice: " + (self.core.voice.status if self.core.voice else "unavailable")
                            + "\nTTS: " + self.core.tts.status),
            font_size=sp(8), color=_rgba(self.palette["dim"]),
        )
        self._row(self.voice_status, 58, T("Status", "الحالة"))

        self._section(T("Privacy & security", "الخصوصية والأمان"))
        self._switch_row(T("Confirm high-risk actions", "تأكيد الإجراءات الخطرة"), cfg.confirm_high_risk, self.set_confirm)
        self._switch_row(T("Allow installed Python skill entry points", "السماح بنقاط دخول بايثون"), cfg.allow_python_skills, self.set_python_skills)
        retention = _shaped_input(self.palette, text=str(cfg.retention_days), input_filter="int",
                                  size_hint=(1, 1), hint_text=T("0 = forever", "0 = دائم"))
        apply_btn = NeonButton(self.palette, text=T("Apply", "تطبيق"), size_hint=(None, 1), width=dp(44), font_size=sp(8))
        apply_btn.bind(on_release=lambda *_: self.set_retention(retention.text))
        wrap2 = BoxLayout(spacing=dp(3))
        wrap2.add_widget(retention)
        wrap2.add_widget(apply_btn)
        self._row(wrap2, 38, T("Retention (days)", "الاحتفاظ (أيام)"))
        self._row(
            WrapLabel(text=shape_text("Encryption: " + self.core.store.crypto.status),
                      font_size=sp(8), color=_rgba(self.palette["good"] if self.core.store.crypto.available else self.palette["danger"])),
            46, T("At rest", "عند التخزين"),
        )

        self._section(T("Intelligence surfaces", "أسطح الذكاء"))
        self._switch_row(T("Proactive morning/evening briefs", "الموجز الاستباقي صباحاً ومساءً"),
                         cfg.proactive_enabled, self.set_proactive)
        self._switch_row(T("Detect commitments in chat", "التقاط الالتزامات من المحادثة"),
                         cfg.commitment_detection, self.set_commitment_detection)
        self._switch_row(T("Auto-compact long conversations", "ضغط المحادثات الطويلة تلقائياً"),
                         cfg.compaction_enabled, self.set_compaction)
        self._chip_row(
            T("Morning brief hour", "ساعة موجز الصباح"),
            [(str(hour), f"{hour:02d}:00") for hour in range(5, 12)],
            str(cfg.morning_hour), self.set_morning_hour,
        )
        self._chip_row(
            T("Evening review hour", "ساعة مراجعة المساء"),
            [(str(hour), f"{hour:02d}:00") for hour in range(17, 24)],
            str(cfg.evening_hour), self.set_evening_hour,
        )
        quiet_row = BoxLayout(spacing=dp(3))
        quiet_start = _shaped_input(self.palette, text=cfg.quiet_start, size_hint=(1, 1), hint_text="22:00")
        quiet_end = _shaped_input(self.palette, text=cfg.quiet_end, size_hint=(1, 1), hint_text="07:00")
        quiet_apply = NeonButton(self.palette, text=T("Set", "ضبط"), size_hint=(None, 1), width=dp(42), font_size=sp(8))
        quiet_apply.bind(on_release=lambda *_: self.set_quiet_hours(quiet_start.text, quiet_end.text))
        quiet_row.add_widget(quiet_start)
        quiet_row.add_widget(quiet_end)
        quiet_row.add_widget(quiet_apply)
        self._row(quiet_row, 38, T("Quiet hours", "ساعات الهدوء"))
        self._row(
            WrapLabel(text=shape_text(f"Commitments: {len(self.core.proactive.open_commitments(20))} open · "
                                      f"digests: {self.core.store.query_one('SELECT COUNT(*) AS c FROM digests')['c']}"),
                      font_size=sp(8), color=_rgba(self.palette["dim"])),
            36, T("Follow-ups", "المتابعات"),
        )
        self._switch_row(T("Mini floating assistant", "المساعد العائم المصغّر"),
                         cfg.float_enabled, self.set_float)
        for text, callback in [
            (T("Open encrypted backups", "افتح النسخ المشفرة"), self.open_backups),
            (T("Rebuild recall index", "أعد بناء فهرس الاسترجاع"), self.reindex_recall),
            (T("Compact this conversation now", "اضغط هذه المحادثة الآن"), self.compact_now),
            (T("Plan mode", "وضع الخطة"), lambda: self.app.show_plan_modal()),
        ]:
            button = NeonButton(self.palette, text=text, size_hint_y=None, height=dp(34))
            button.bind(on_release=lambda *_args, cb=callback: self.app.ui_guard(cb))
            self.box.add_widget(button)
        backup_days = _shaped_input(self.palette, text=str(cfg.auto_backup_days), input_filter="int",
                                    size_hint=(1, 1), hint_text=T("0 = off", "0 = متوقف"))
        backup_apply = NeonButton(self.palette, text=T("Apply", "تطبيق"), size_hint=(None, 1), width=dp(44), font_size=sp(8))
        backup_apply.bind(on_release=lambda *_: self.set_auto_backup(backup_days.text))
        backup_wrap = BoxLayout(spacing=dp(3))
        backup_wrap.add_widget(backup_days)
        backup_wrap.add_widget(backup_apply)
        self._row(backup_wrap, 38, T("Auto-backup (days)", "نسخ تلقائي (أيام)"))

        self._section(T("Arabic-first", "العربية أولاً"))
        self._switch_row(T("Hijri date in the header", "التاريخ الهجري في الترويسة"), cfg.hijri_dates, self.set_hijri)
        self._switch_row(T("Arabic-Indic numerals", "الأرقام العربية الهندية"), cfg.arabic_numerals, self.set_arabic_numerals)
        self._switch_row(T("Right-to-left shell", "واجهة من اليمين لليسار"), cfg.rtl_layout, self.set_rtl)

        self._section(T("Local data", "البيانات المحلية"))
        for text, callback in [
            (T("Clear conversations", "مسح المحادثات"), self.clear_conversations),
            (T("Clear all memories", "مسح كل الذكريات"), self.clear_memories),
            (T("Reset all local data", "إعادة ضبط كل البيانات"), self.reset_all),
        ]:
            button = NeonButton(self.palette, text=text, tone="danger" if "Reset" in text or "ضبط" in text else "accent",
                                size_hint_y=None, height=dp(34))
            button.bind(on_release=lambda *_args, cb=callback: self.app.ui_guard(cb))
            self.box.add_widget(button)

        self._section(T("About", "حول"))
        self.box.add_widget(WrapLabel(
            text=shape_text(
                f"{APP_NAME} v{APP_VERSION} · build {BUILD_STAMP}\n"
                f"private: {PRIVATE_DIR}\nshared: {PUBLIC_DIR}\nerrors: {ERROR_LOG_PATH}\n"
                "Offline-first. No telemetry leaves the device. Online tools run only when you invoke them."
            ),
            font_size=sp(7.8), color=_rgba(self.palette["dim"]),
        ))

    # -- setters ---------------------------------------------------------
    def set_ui_lang(self, value: str) -> None:
        self.core.config.set("ui_lang", value)
        set_ui_lang(value)
        self.app.rebuild_ui()

    def set_animations(self, value: bool) -> None:
        self.core.config.set("animations", value)
        if self.app.background:
            self.app.background._active = bool(value)

    def set_ui_sound(self, value: bool) -> None:
        self.core.config.set("ui_sound", bool(value))
        if value:
            self.core.ui_sound.play("toggle_on")

    def set_ui_volume(self, value: float) -> None:
        self.core.config.set("ui_sound_volume", clamp(float(value), 0.0, 1.0))
        self.core.ui_sound.play("tap")

    def test_sound(self) -> None:
        if self.core.ui_sound.test():
            self.app.toast(T("Test tone sent to the audio output.", "أُرسلت نغمة الاختبار إلى مخرج الصوت."))
        else:
            self.app.toast(T(
                "No audio output is available. Check media volume and re-enable button sounds.",
                "لا يوجد مخرج صوت متاح. تحقق من مستوى الصوت وأعد تفعيل أصوات الأزرار.",
            ), 5)

    def set_intensity(self, value: float) -> None:
        self.core.config.set("shader_intensity", value)
        if self.app.background:
            self.app.background.intensity = float(value)

    def set_button_style(self, value: str) -> None:
        self.core.config.set("button_style", value)
        self.app.rebuild_ui()

    def set_sphere_style(self, value: str) -> None:
        self.core.config.set("sphere_style", value)
        if self.app.background and value in VALID_SPHERE_STYLES:
            self.app.background.sphere_style = float(VALID_SPHERE_STYLES.index(value))
        if getattr(self.app, "_float_running", False):
            self.core.bridge.float_refresh(self.core.config.sphere_color, value)

    def set_sphere_color(self, value: str) -> None:
        if value not in VALID_SPHERE_COLORS:
            value = "cyan"
        self.core.config.set("sphere_color", value)
        primary, glow = sphere_colors(value)
        if self.app.background:
            self.app.background.sphere_tint = primary
            self.app.background.sphere_glow = glow
        orb = getattr(self.app, "float_orb", None)
        if orb is not None:
            orb.apply_sphere_colors(value)
        if getattr(self.app, "_float_running", False):
            self.core.bridge.float_refresh(value, self.core.config.sphere_style)

    def _asset_section(self) -> None:
        self._section(T("Local models (assets folder)", "النماذج المحلية (مجلد الأصول)"))
        self.asset_dots: Dict[str, Any] = {}
        try:
            rows = build_asset_status(self.core)
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"asset status failed: {exc}")
            rows = []
        for row in rows:
            panel = BoxLayout(size_hint_y=None, height=dp(30), spacing=dp(3))
            dot = _label("●" if row["ok"] else "○", self.palette, 11.0,
                         "good" if row["ok"] else "danger", size_hint=(None, 1), width=dp(15))
            panel.add_widget(dot)
            panel.add_widget(_label(row["label"], self.palette, 8.8, "text", size_hint_y=1))
            if row["key"] in ASSET_NAMES:
                get_btn = NeonButton(self.palette, text=T1("Download", "تنزيل"),
                                     size_hint=(None, 1), width=dp(64), font_size=sp(8))
                get_btn.bind(on_release=lambda *_a, key=row["key"], label=row["label"]:
                             self.download_asset(key, label))
                panel.add_widget(get_btn)
            self.box.add_widget(panel)
            self.asset_dots[row["key"]] = dot
        folder = self.core.config.assets_path or str(ASSETS_DIR)
        self.box.add_widget(_label(
            folder, self.palette, 7.4, "dim", shorten=True,
            height=dp(16), text_size=(dp(320), dp(16)),
        ))
        open_btn = NeonButton(self.palette, text=T("Open model manager", "فتح مدير النماذج"),
                              size_hint_y=None, height=dp(34))
        open_btn.bind(on_release=lambda *_: self.app.ui_guard(lambda: self.app.show_screen("models")))
        self.box.add_widget(open_btn)

    def download_asset(self, key: str, label: str) -> None:
        models = self.app.ensure_screen("models")
        self.app.show_screen("models")
        Clock.schedule_once(lambda _dt: models.ask_download(key, label), 0.05)

    def refresh(self) -> None:
        status = getattr(self, "voice_status", None)
        if status is not None:
            with contextlib.suppress(Exception):
                status.text = shape_text(
                    "Voice: " + (self.core.voice.status if self.core.voice else "unavailable")
                    + "\nTTS: " + self.core.tts.status
                )
        try:
            rows = build_asset_status(self.core)
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"settings asset refresh failed: {exc}")
            return
        for row in rows:
            dot = getattr(self, "asset_dots", {}).get(row["key"])
            if dot is None:
                continue
            dot.text = "●" if row["ok"] else "○"
            dot.color = _rgba(self.palette["good"] if row["ok"] else self.palette["danger"])

    def set_profile(self, value: str) -> None:
        self.core.config.set("profile", value)
        self.app.toast(T("Profile saved. Reload model if loaded.", "تم حفظ الملف. أعد تحميل النموذج إن كان محملاً."))

    def set_auto_unload(self, value: bool) -> None:
        self.core.config.set("auto_unload_model", value)

    def set_online_mode(self, value: str) -> None:
        self.core.config.set("online_mode", value)
        self.app.update_chrome_status()

    def set_language(self, value: str) -> None:
        self.core.config.set("language", value)

    def set_tts(self, value: bool) -> None:
        self.core.config.set("tts_enabled", value)
        if not value:
            self.core.tts.stop()

    def set_tts_rate(self, value: float) -> None:
        self.core.config.set("tts_rate", round(float(value), 2))

    def set_tts_voice(self, value: str) -> None:
        self.core.config.set("tts_voice", value)
        self.app.run_bg(self.core.reconfigure_tts, lambda _r: self.app.toast(T("TTS backend reloaded", "تم إعادة تحميل محرك النطق")))

    def choose_voice(self) -> None:
        self.app.show_screen("models")
        screen = self.app.ensure_screen("models")
        scroll = getattr(screen, "scroll_to_voices", None)
        if callable(scroll):
            scroll()

    def set_wake(self, value: str) -> None:
        value = (value or "").strip()[:48]
        if value:
            self.core.config.set("wake_word", value)
            self.app.toast(T("Wake word saved", "تم حفظ كلمة التنبيه"))

    def set_continuous(self, value: bool) -> None:
        self.core.config.set("continuous_conversation", value)

    def set_confirm(self, value: bool) -> None:
        self.core.config.set("confirm_high_risk", value)

    def set_python_skills(self, value: bool) -> None:
        self.core.config.set("allow_python_skills", value)
        self.app.toast(T(
            "Python skills run in a separate process with path/permission gates; this is not a full sandbox.",
            "تعمل مهارات بايثون في عملية منفصلة مع قيود المسارات والأذونات؛ هذا ليس صندوقاً عازلاً كاملاً.",
        ))

    def set_retention(self, value: str) -> None:
        try:
            days = max(0, min(int(value or 0), 3650))
        except ValueError:
            days = 0
        self.core.config.set("retention_days", days)
        self.app.toast(T("Retention updated", "تم تحديث الاحتفاظ"))

    def set_proactive(self, value: bool) -> None:
        self.core.config.set("proactive_enabled", bool(value))

    def set_commitment_detection(self, value: bool) -> None:
        self.core.config.set("commitment_detection", bool(value))

    def set_compaction(self, value: bool) -> None:
        self.core.config.set("compaction_enabled", bool(value))

    def set_morning_hour(self, value: str) -> None:
        with contextlib.suppress(ValueError):
            self.core.config.set("morning_hour", max(0, min(int(value), 23)))

    def set_evening_hour(self, value: str) -> None:
        with contextlib.suppress(ValueError):
            self.core.config.set("evening_hour", max(0, min(int(value), 23)))

    def set_quiet_hours(self, start: str, end: str) -> None:
        self.core.config.set("quiet_start", (start or "22:00").strip())
        self.core.config.set("quiet_end", (end or "07:00").strip())
        self.app.toast(T("Quiet hours saved", "حُفظت ساعات الهدوء"), 2.2)

    def set_hijri(self, value: bool) -> None:
        self.core.config.set("hijri_dates", bool(value))
        self.app.tick(0)

    def set_arabic_numerals(self, value: bool) -> None:
        self.core.config.set("arabic_numerals", bool(value))
        self.app.tick(0)

    def set_rtl(self, value: bool) -> None:
        self.core.config.set("rtl_layout", bool(value))
        self.app.toast(T("Layout direction applies after a UI rebuild.",
                         "يُطبّق اتجاه الواجهة بعد إعادة البناء."), 3)
        self.app.rebuild_ui()

    def set_float(self, value: bool) -> None:
        enabled = bool(value)
        self.core.config.set("float_enabled", enabled)
        self.app._mini_visible = enabled
        if enabled:
            self.core.bridge.float_clear_suppressed()
        if IS_ANDROID and enabled:
            status = self.core.bridge.float_status()
            if not status.get("granted") and not status.get("running"):
                self.core.bridge.request_overlay_permission()
                self.app._apply_orb_visibility()
                self.app._place_orb(initial=True)
                self.app.toast(T(
                    "Allow “display over other apps”, then enable Float again.",
                    "اسمح بـ«العرض فوق التطبيقات الأخرى»، ثم فعّل العائم مجدداً.",
                ), 6)
                return
        self.app._sync_float()
        self.app._place_orb(initial=True)
        if enabled and IS_ANDROID and not self.app._float_running:
            self.app.toast(T("Float could not start. The in-app orb is shown instead.",
                             "تعذّر تشغيل العائم. يظهر الزر داخل التطبيق بدلاً منه."), 5)
        elif enabled:
            self.app.toast(T("Floating assistant is on.", "المساعد العائم يعمل."), 3)

    def set_auto_backup(self, value: str) -> None:
        with contextlib.suppress(ValueError):
            self.core.config.set("auto_backup_days", max(0, min(int(value or 0), 30)))
        self.app.toast(T("Automatic backup schedule saved", "حُفظ جدول النسخ التلقائي"), 2.4)

    def open_backups(self) -> None:
        self.app.show_backup_modal()

    def reindex_recall(self) -> None:
        self.app.toast(T("Rebuilding the recall index…", "جارٍ إعادة بناء الفهرس…"), 2.4)
        self.app.run_bg(
            self.core.recall.reindex,
            lambda counts: self.app.toast(T(
                f"Recall index rebuilt · {sum((counts or {}).values())} items",
                f"أُعيد بناء الفهرس · {sum((counts or {}).values())} عنصر",
            ), 3.5),
        )

    def compact_now(self) -> None:
        def work() -> Dict[str, Any]:
            return self.core.compactor.compact(force=True)

        def done(outcome: Dict[str, Any]) -> None:
            if outcome.get("ok"):
                self.app.toast(T(
                    f"Compacted {outcome.get('count', 0)} earlier turns",
                    f"ضُغطت {outcome.get('count', 0)} أدوار سابقة",
                ), 3)
            else:
                self.app.toast(str(outcome.get("error") or "Nothing to compact"), 3)

        self.app.run_bg(work, done)

    def about(self) -> None:
        modal = BoshraModal(self.app, T("About Boshra AI", "حول بشرى"), self.palette)
        modal.text(
            f"{APP_NAME} {APP_VERSION}\n{BUILD_STAMP}\n"
            + T("Sovereign offline intelligence. Local models, local memory, local skills.",
                "ذكاء سيادي دون اتصال. نماذج محلية، ذاكرة محلية، مهارات محلية."),
            10,
        )
        modal.text(
            T("Privacy: no analytics, no background cloud calls, no hidden recording. Audio capture runs only while the visible microphone control is active.",
              "الخصوصية: بلا تحليلات أو اتصالات سحابية خفية أو تسجيل مخفي. يعمل الصوت فقط أثناء تفعيل زر الميكروفون الظاهر."),
            8.5, "dim",
        )

    # -- destructive data actions ----------------------------------------
    def _confirmed(self, question: str, detail: str, action: Callable[[], Any]) -> None:
        def confirmed(answer: bool) -> None:
            if answer:
                self.app.run_bg(action, lambda result: (self.app.toast(str((result or {}).get("message", "Done"))), self.app.rebuild_ui()))
        self.app.confirm_async(question, detail, confirmed)

    def clear_conversations(self) -> None:
        def work() -> Any:
            self.core.store.execute("DELETE FROM messages")
            self.core.store.execute("DELETE FROM conversations")
            self.core.config.set("last_conversation_id", 0)
            self.core.store.audit("privacy", "clear-conversations", "", "high")
            return {"message": "Conversations cleared"}
        self._confirmed(T("Delete all conversations?", "حذف كل المحادثات؟"), T("This cannot be undone.", "لا يمكن التراجع."), work)

    def clear_memories(self) -> None:
        def work() -> Any:
            count = self.core.memory.clear()
            return {"message": f"Cleared {count} memories"}
        self._confirmed(T("Delete all memories?", "حذف كل الذكريات؟"), T("Preferences, facts and project memory will be erased.", "ستُمسح التفضيلات والحقائق وذاكرة المشاريع."), work)

    def reset_all(self) -> None:
        self._confirmed(
            T("Reset ALL local data?", "إعادة ضبط كل البيانات المحلية؟"),
            T("Conversations, memories, vault index, skills, routines and local settings are erased. Original imported files are untouched.",
              "ستُمسح المحادثات والذاكرة وفهرس الخزنة والمهارات والأتمتة والإعدادات. الملفات الأصلية لن تمس."),
            self.core.reset_all_data,
        )


SCREEN_CLASSES: Dict[str, Any] = {
    "chat": ChatScreen,
    "skills": SkillsScreen,
    "memory": MemoryScreen,
    "vault": VaultScreen,
    "flow": AutomationScreen,
    "sos": EmergencyScreen,
    "privacy": PrivacyScreen,
    "diag": DiagnosticsScreen,
    "models": ModelsScreen,
    "settings": SettingsScreen,
}

NAV_ITEMS = [
    ("chat", "Chat", "محادثة"),
    ("skills", "Skills", "مهارات"),
    ("memory", "Memory", "ذاكرة"),
    ("vault", "Vault", "خزنة"),
    ("flow", "Flows", "أتمتة"),
    ("sos", "SOS", "طوارئ"),
    ("privacy", "Privacy", "خصوصية"),
    ("diag", "Diag", "تشخيص"),
    ("models", "Models", "نماذج"),
    ("settings", "Set", "إعداد"),
]


# ==========================================================================
# Bootstrapping: service container
# ==========================================================================
class BoshraCore:
    """Builds every service once and wires the shared AppContext."""

    def __init__(self) -> None:
        self._shutting_down = False
        self.app: Optional[Any] = None
        self.busy = False
        self._cpu_prev: Optional[Tuple[float, float]] = None
        self._cpu_value = 0.0
        self._cpu_at = 0.0

        self.config = ConfigManager(CONFIG_PATH)
        try:
            self.store = Store(DB_PATH, encrypt=bool(self.config.encrypt_storage))
        except Exception as exc:  # noqa: BLE001
            LOG.error("database open failed; preserving the original file", exc)
            _append_error_dump("database-open", traceback.format_exc())
            if DB_PATH.exists():
                backup = DB_PATH.with_name(DB_PATH.name + f".failed-{int(time.time())}")
                with contextlib.suppress(OSError):
                    shutil.move(str(DB_PATH), str(backup))
            self.store = Store(DB_PATH, encrypt=False)

        self.memory = MemoryManager(self.store)
        self.vault = KnowledgeVault(self.store)
        self.bridge = AndroidBridge()
        self.ui_sound = UISound(self.config, self.bridge)
        set_ui_sound_service(self.ui_sound)
        self.resources = ResourceManager(self.bridge, self.config)
        self.models = ModelManager(self.config, self.store)
        self.llm = LLMService(self.config, self.models, self.resources, self.store)

        ctx = AppContext()
        self.ctx = ctx
        ctx.store = self.store
        ctx.config = self.config
        ctx.memory = self.memory
        ctx.vault = self.vault
        ctx.bridge = self.bridge
        ctx.resources = self.resources
        ctx.models = self.models
        ctx.llm = self.llm
        ctx.ui_emit = lambda event, payload: None

        self.permissions = PermissionManager(self.store, self.config)
        ctx.permissions = self.permissions

        self.diagnostics = DiagnosticsEngine(ctx)
        ctx.diagnostics = self.diagnostics
        self.emergency = EmergencyGuide(self.config)
        ctx.emergency = self.emergency

        self.tools = build_tool_registry(ctx)
        ctx.tools = self.tools

        self.skills = SkillEngine(ctx)
        ctx.skills = self.skills

        self.automation = WorkflowEngine(ctx)
        ctx.automation = self.automation

        self.recall = RecallIndex(ctx)
        ctx.recall = self.recall
        self.backups = BackupManager(ctx)
        ctx.backups = self.backups
        self.planner = PlanEngine(ctx)
        ctx.planner = self.planner
        self.proactive = ProactiveEngine(ctx)
        ctx.proactive = self.proactive
        self.compactor = ConversationCompactor(ctx)
        ctx.compactor = self.compactor

        self.tts = SpeechEngine(self.bridge, self.config, on_state=self._on_tts_state)
        ctx.tts = self.tts

        self.voice = VoiceEngine(
            self.bridge, self.config, self.resources, self.store,
            on_state=self._on_voice_state,
            on_partial=self._on_voice_partial,
            on_final=self._on_voice_final,
            on_error=self._on_voice_error,
        )
        ctx.voice = self.voice

        self.orchestrator = IntelligenceOrchestrator(ctx)
        ctx.orchestrator = self.orchestrator
        ctx.confirm_handler = self._confirm
        self.permissions.confirm_handler = self._confirm

        stored_focus = self.store.state_get("focus_end", "")
        with contextlib.suppress(ValueError, TypeError):
            ctx.focus_end = float(stored_focus or 0.0)
        self.store.audit("system", "start", f"v{APP_VERSION}", "low")

    # -- notification bridges -------------------------------------------
    def _on_tts_state(self, state: str) -> None:
        self.ctx.emit("tts", state=state)

    def _on_voice_state(self, state: str) -> None:
        self.ctx.emit("voice_state", state=state)

    def _on_voice_partial(self, text: str) -> None:
        self.ctx.emit("voice_partial", text=text)

    def _on_voice_final(self, text: str, lang: str) -> None:
        self.ctx.emit("voice_final", text=text, lang=lang)

    def _on_voice_error(self, text: str) -> None:
        self.ctx.emit("voice_error", text=text)

    def _confirm(self, question: str, detail: str = "") -> bool:
        app = self.app
        handler = getattr(app, "confirm_sync", None)
        if callable(handler):
            return bool(handler(question, detail))
        LOG.warning("confirmation required but no application UI is attached")
        return False

    # -- orchestration ---------------------------------------------------
    def process_text(
        self,
        text: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_state: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        return self.orchestrator.process(text, on_token=on_token, on_state=on_state, on_done=on_done)

    def reconfigure_tts(self) -> Dict[str, Any]:
        old = self.tts
        with contextlib.suppress(Exception):
            old.stop()
            old.shutdown()
        self.tts = SpeechEngine(self.bridge, self.config, on_state=self._on_tts_state)
        self.ctx.tts = self.tts
        self.store.audit("voice", "tts-reconfigure", self.tts.status, "low")
        return {"ok": True, "message": f"Speech backend: {self.tts.status}"}

    # -- telemetry -------------------------------------------------------
    def cpu_percent(self) -> float:
        now = time.time()
        if now - self._cpu_at < 1.0 and self._cpu_prev is not None:
            return self._cpu_value
        try:
            with open("/proc/stat", "r", encoding="ascii", errors="ignore") as handle:
                fields = handle.readline().split()
            values = [float(value) for value in fields[1:]]
            if len(values) < 4:
                raise ValueError("malformed /proc/stat")
            idle = values[3] + (values[4] if len(values) > 4 else 0.0)
            total = sum(values)
            if self._cpu_prev is not None:
                delta_total = total - self._cpu_prev[0]
                delta_idle = idle - self._cpu_prev[1]
                if delta_total > 0:
                    self._cpu_value = clamp(100.0 * (1.0 - delta_idle / delta_total), 0.0, 100.0)
            self._cpu_prev = (total, idle)
            self._cpu_at = now
        except (OSError, ValueError, IndexError):
            self._cpu_at = now
        return self._cpu_value

    def telemetry(self) -> Dict[str, Any]:
        data = dict(self.resources.measure())
        data["cpu"] = self.cpu_percent()
        return data

    # -- maintenance -----------------------------------------------------
    @staticmethod
    def _clear_dir(path: Path) -> int:
        removed = 0
        if not path.exists():
            return 0
        for entry in list(path.iterdir()):
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed += 1
            except OSError as exc:
                LOG.warning(f"could not remove {entry}: {exc}")
        return removed

    def reset_all_data(self) -> Dict[str, Any]:
        self.orchestrator.stop_event.set()
        with contextlib.suppress(Exception):
            self.automation.cancel()
            self.automation.reset()
        with contextlib.suppress(Exception):
            if self.voice is not None:
                self.voice.stop()
        with contextlib.suppress(Exception):
            self.tts.stop()
        with contextlib.suppress(Exception):
            self.llm.unload()

        tables = (
            "messages", "memory_vectors", "chunks", "conversations", "memories",
            "documents", "skill_grants", "skill_history", "perms", "audit",
            "tasks", "notes", "events", "reminders", "routines", "app_state",
        )
        for table in tables:
            self.store.execute(f"DELETE FROM {table}")
        self.store.execute("DELETE FROM skills WHERE source!='builtin'")
        with contextlib.suppress(Exception):
            self.store.execute("DELETE FROM sqlite_sequence")
        self.store.vacuum()

        for directory in (SKILLS_DIR, self.vault.vault_files, EXPORT_DIR, PRIVATE_DIR / "cache"):
            self._clear_dir(directory)
            with contextlib.suppress(OSError):
                directory.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            self.ui_sound.reset()

        self.ctx.recording_steps.clear()
        self.ctx.pending = None
        self.ctx.focus_end = 0.0
        self.config.data = Config()
        self.config.data.encrypt_storage = bool(self.store.crypto.available)
        self.config.save()
        set_ui_lang(self.config.data.ui_lang)
        with contextlib.suppress(Exception):
            self.skills.discover()
        with contextlib.suppress(Exception):
            self.automation._ensure_presets()
        self.memory = MemoryManager(self.store)
        self.ctx.memory = self.memory
        self.vault = KnowledgeVault(self.store)
        self.ctx.vault = self.vault
        self.busy = False
        self.store.audit("privacy", "reset-all", "local data erased", "high")
        return {"ok": True, "message": "All local data was reset"}

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        with contextlib.suppress(Exception):
            self.orchestrator.stop_event.set()
        with contextlib.suppress(Exception):
            self.automation.cancel()
        with contextlib.suppress(Exception):
            if self.voice is not None:
                self.voice.stop()
        with contextlib.suppress(Exception):
            self.tts.stop()
            self.tts.shutdown()
        with contextlib.suppress(Exception):
            self.llm.unload()
        with contextlib.suppress(Exception):
            self.store.audit("system", "stop", "", "low")
        with contextlib.suppress(Exception):
            self.store.close()


# ==========================================================================
# Application shell
# ==========================================================================
STATE_LABELS: Dict[str, Tuple[str, str]] = {
    "idle": ("Ready", "جاهز"),
    "thinking": ("Thinking", "أفكر"),
    "speaking": ("Speaking", "أتحدث"),
    "listening": ("Listening", "أستمع"),
}

STATE_VALUES = {"idle": 0.0, "thinking": 1.0, "speaking": 2.0, "listening": 3.0}
STATE_ENERGY = {"idle": 0.62, "thinking": 1.05, "speaking": 1.24, "listening": 0.95}
SCREEN_ALIASES = {
    "home": "chat", "chat": "chat", "skills": "skills", "memory": "memory",
    "vault": "vault", "flow": "flow", "automation": "flow", "sos": "sos",
    "emergency": "sos", "privacy": "privacy", "diag": "diag",
    "diagnostics": "diag", "models": "models", "settings": "settings",
}


class BoshraApp(App):
    """Android-first Kivy shell: one GLSL aurora pass plus glass dashboards."""

    title = f"{APP_NAME} {APP_VERSION}"

    def build(self) -> Any:
        register_fonts()
        # Validate bundled assets before any screen is composed. Every failure
        # path keeps the UI on Kivy/system defaults instead of raising.
        self._startup_report: Optional[AssetReport] = None
        self._asset_capabilities: Dict[str, bool] = {}
        self._startup_asset_guard()
        Window.softinput_mode = "resize"
        if IS_ANDROID:
            with contextlib.suppress(Exception):
                Window.fullscreen = "auto"
        else:
            with contextlib.suppress(Exception):
                screen_width, screen_height = Window.system_size
                width_env = os.environ.get("BOSHRA_WIDTH")
                height_env = os.environ.get("BOSHRA_HEIGHT")
                width = int(width_env) if width_env else min(520, max(360, int(screen_width * 0.40)))
                height = int(height_env) if height_env else min(960, max(560, int(screen_height * 0.92)))
                Window.size = (max(320, width), max(520, height))

        self._shutting_down = False
        self._rebuild_scheduled = False
        self._post_rebuild_toast = ""
        self._toast_event: Optional[Any] = None
        self._tick_event: Optional[Any] = None
        self._accel_event: Optional[Any] = None
        self._accel = None
        self._tick_counter = 0
        self._last_reminder_scan = 0.0
        self._last_maintenance = 0.0
        self._tts_speaking = False
        self._smoke_mode = os.environ.get("BOSHRA_SMOKE") == "1"
        self.modals: List[Any] = []
        self.screens: Dict[str, Any] = {}
        self.nav_buttons: Dict[str, Any] = {}
        self.current_screen = "chat"
        self.telem_enabled = True

        self.core = BoshraCore()
        self.core.app = self
        self.core.ctx.ui_emit = self._ui_event
        self.core.ctx.confirm_handler = self.confirm_sync
        self.core.permissions.confirm_handler = self.confirm_sync
        self.palette = PALETTES.get(self.core.config.theme, PALETTES["aurora"])
        global BUTTON_STYLE
        BUTTON_STYLE = self.core.config.button_style
        self._refresh_insets()
        self._float_state = self.core.bridge.float_status()
        self._float_running = bool(self._float_state.get("running"))
        self._insets_event = Clock.schedule_interval(self._refresh_insets_safe, 6.0)

        self._root = FloatLayout()
        self._compose()
        self._root.bind(size=self.update_layout)
        with contextlib.suppress(Exception):
            Window.bind(on_keyboard=self._on_keyboard)
        Clock.schedule_once(self._after_start, 0.15)
        return self._root

    def _startup_asset_guard(self) -> None:
        """One defensive scan at launch; never blocks or crashes the UI.

        Missing folders are simply skipped: the chat, voice and font layers
        each fall back to Android/Kivy defaults on their own.
        """
        started = time.monotonic()
        try:
            folders = ensure_asset_dirs()
        except Exception as exc:  # noqa: BLE001 - launch must survive
            LOG.warning(f"asset folder creation skipped: {exc}")
            folders = {}
        try:
            report = inspect_assets()
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"asset scan skipped: {exc}")
            report = None
        self._startup_report = report
        if report is None:
            return
        self._asset_capabilities = {row.key: bool(row.ok) for row in report.statuses}
        missing = [row.key for row in report.statuses if not row.ok]
        elapsed = (time.monotonic() - started) * 1000
        LOG.info(
            f"asset scan {elapsed:.0f} ms · root={report.root} · "
            + ("all present" if not missing else "missing " + ", ".join(missing))
        )
        for key in missing:
            entry = folders.get(key)
            LOG.warning(f"asset '{key}' unavailable; using system fallback"
                        + (f" (expected in {entry})" if entry else ""))

    # -- composition -----------------------------------------------------
    def _compose(self) -> None:
        self._cancel_timers()
        for modal in list(self.modals):
            with contextlib.suppress(Exception):
                modal.dismiss()
        self.modals.clear()
        self._root.clear_widgets()
        self.screens = {}
        self.nav_buttons = {}
        self.toast_label = None
        self.toast_panel = None
        self.telem_labels = {}

        self.background = AuroraBackground(self.palette, size_hint=(None, None))
        self.background.fps = int(self.core.resources.profile.get("shader_fps", 40))
        self.background.intensity = float(self.core.config.shader_intensity)
        self.background._active = bool(self.core.config.animations)
        with contextlib.suppress(ValueError):
            self.background.sphere_style = float(
                VALID_SPHERE_STYLES.index(self.core.config.sphere_style)
            )
        sphere_primary, sphere_glow = sphere_colors(self.core.config.sphere_color)
        self.background.sphere_tint = sphere_primary
        self.background.sphere_glow = sphere_glow
        self._root.add_widget(self.background)

        self.sphere_touch = SphereTouch(self, size_hint=(None, None))
        self._root.add_widget(self.sphere_touch)

        self.telemetry = GlassPanel(
            self.palette, orientation="vertical", size_hint=(None, None),
            alpha=0.55, radius=10, padding=[dp(5), dp(4)], spacing=dp(1),
            border_alpha=0.18,
        )
        self.telemetry.add_widget(_label("SYSTEM", self.palette, 7.6, "accent", bold=True))
        for key in ("ram", "cpu", "bat", "temp", "free", "net", "cores", "uptime"):
            label = _label(f"{key.upper():<5} --", self.palette, 7.0, "dim")
            self.telemetry.add_widget(label)
            self.telem_labels[key] = label
        self._root.add_widget(self.telemetry)

        self.host = BoxLayout(
            orientation="vertical", size_hint=(None, None),
            spacing=0, padding=0,
        )
        self._root.add_widget(self.host)

        self.header = GlassPanel(
            self.palette, orientation="horizontal", size_hint=(None, None),
            alpha=0.74, radius=0, padding=[dp(6), dp(3), dp(9), dp(3)], spacing=dp(3),
            border=True, border_alpha=0.16,
        )
        brand = BoxLayout(orientation="vertical", size_hint_x=0.24, spacing=0)
        brand.add_widget(_label(APP_NAME.upper(), self.palette, 9.6, "accent", bold=True, shorten=True))
        brand.add_widget(_label(f"v{APP_VERSION}", self.palette, 6.4, "dim"))
        self.header.add_widget(brand)

        clock_box = BoxLayout(orientation="vertical", size_hint_x=0.42, spacing=0)
        self.clock_label = _label("--:--", self.palette, 15.5, "text", bold=True, halign="center")
        self.date_label = _label("", self.palette, 6.8, "dim", halign="center")
        clock_box.add_widget(self.clock_label)
        clock_box.add_widget(self.date_label)
        self.header.add_widget(clock_box)

        state_box = BoxLayout(orientation="vertical", size_hint_x=0.26, spacing=0)
        self.state_label = _label(STATE_LABELS["idle"][0], self.palette, 7.6, "accent", bold=True, halign="right")
        self.status_label = _label("", self.palette, 6.3, "dim", halign="right", shorten=True)
        state_box.add_widget(self.state_label)
        state_box.add_widget(self.status_label)
        self.header.add_widget(state_box)

        telemetry_toggle = NeonButton(
            self.palette, text="TEL", size_hint=(None, 1), width=dp(30), font_size=sp(8.5),
            bold=True, halign="center",
        )
        telemetry_toggle.bind(on_release=lambda *_: self.ui_guard(self.toggle_telemetry))
        self.header.add_widget(telemetry_toggle)
        if bool(self.core.config.rtl_layout) and _UI_LANG == "ar":
            children = list(self.header.children)
            self.header.clear_widgets()
            for child in reversed(children):
                self.header.add_widget(child)
        self._root.add_widget(self.header)

        self.nav_grid = BoxLayout(
            orientation="vertical", size_hint=(None, None),
            spacing=dp(2), padding=[dp(2), dp(1)],
        )
        nav_rows = []
        for row_index in range(2):
            row = BoxLayout(orientation="horizontal", spacing=dp(3), size_hint_y=1)
            nav_rows.append(row)
            self.nav_grid.add_widget(row)
        for index, (key, en, ar) in enumerate(NAV_ITEMS):
            if _UI_LANG == "ar":
                text = ar
            elif _UI_LANG == "en":
                text = en
            else:
                text = f"{en}\n{ar}"
            button = NeonButton(
                self.palette, text=text, size_hint=(1, 1),
                font_size=sp(7.4), alpha=0.12,
            )
            button.bind(on_release=lambda *_args, nav_key=key: self.ui_guard(lambda: self.show_screen(nav_key)))
            nav_rows[index // 5].add_widget(button)
            self.nav_buttons[key] = button
        if bool(self.core.config.rtl_layout) and _UI_LANG == "ar":
            for row in nav_rows:
                children = list(row.children)
                row.clear_widgets()
                for child in reversed(children):
                    row.add_widget(child)
        self._root.add_widget(self.nav_grid)

        self.toast_panel = GlassPanel(
            self.palette, orientation="horizontal", size_hint=(None, None),
            alpha=0.88, radius=9, padding=[dp(8), dp(3)], border=True, border_alpha=0.22,
        )
        self.toast_panel.opacity = 0
        self.toast_label = Label(
            text="", font_name=FONT, font_size=sp(8.8), color=_rgba(self.palette["text"]),
            halign="center", valign="middle", shorten=True,
        )
        self.toast_label.bind(size=lambda widget, _value: setattr(widget, "text_size", (widget.width, None)))
        self.toast_panel.add_widget(self.toast_label)
        self._root.add_widget(self.toast_panel)

        self.float_orb = DraggableOrb(self)
        self._root.add_widget(self.float_orb)
        self._mini_visible = bool(self.core.config.float_enabled)
        self._apply_orb_visibility()
        Clock.schedule_once(lambda _dt: self._place_orb(initial=True), 0.05)

        self.show_screen(self.current_screen)
        self.update_layout()
        self._tick_event = Clock.schedule_interval(self.tick, 0.5)
        if self._post_rebuild_toast:
            message, self._post_rebuild_toast = self._post_rebuild_toast, ""
            Clock.schedule_once(lambda _dt: self.toast(message), 0.1)

    def _cancel_timers(self) -> None:
        for attr in ("_tick_event", "_accel_event", "_toast_event", "_insets_event"):
            event = getattr(self, attr, None)
            if event is not None:
                with contextlib.suppress(Exception):
                    event.cancel()
                setattr(self, attr, None)
        accel = getattr(self, "_accel", None)
        if accel is not None:
            with contextlib.suppress(Exception):
                accel.disable()
        background = getattr(self, "background", None)
        if background is not None:
            with contextlib.suppress(Exception):
                background.stop()

    def _after_start(self, _dt: float) -> None:
        chat = self.screens.get("chat")
        if chat is not None:
            self._ensure_welcome(chat)
            chat.refresh_status()
        self.update_chrome_status()
        self.update_telemetry()
        self._start_accelerometer()
        self._maybe_prompt_assets()
        with contextlib.suppress(Exception):
            self.core.ui_sound.warm_up()
        if self.core.config.float_start_on_launch and not self.core.config.float_enabled:
            self.core.config.set("float_enabled", True)
        Clock.schedule_once(lambda _dt: self._sync_float(), 1.0)
        Clock.schedule_once(lambda _dt: self._background_upkeep(), 2.5)
        if self._smoke_mode:
            Clock.schedule_once(self._smoke_capture, 3.0)

    def _background_upkeep(self) -> None:
        def work() -> Dict[str, Any]:
            outcome: Dict[str, Any] = {}
            if self.core.recall is not None:
                outcome["recall"] = self.core.recall.reindex_if_stale()
            if self.core.backups is not None:
                backup = self.core.backups.maybe_auto_backup()
                if backup:
                    outcome["backup"] = backup
            return outcome

        def done(result: Dict[str, Any]) -> None:
            backup = result.get("backup") if isinstance(result, dict) else None
            if backup and backup.get("ok"):
                self.toast(T("Encrypted backup saved.", "حُفظت نسخة احتياطية مشفّرة."), 3)

        self.run_bg(work, done)

    def _sync_float(self) -> None:
        """Reconcile the saved float preference with the live overlay service."""
        config = self.core.config
        desired = bool(config.float_enabled)
        status = self.core.bridge.float_status()
        self._float_state = status
        supported = bool(status.get("supported"))
        granted = bool(status.get("granted"))
        running = bool(status.get("running"))
        if not IS_ANDROID or not supported:
            self._float_running = False
            self._mini_visible = desired
            self._apply_orb_visibility()
            return
        if status.get("suppressed"):
            if desired:
                config.set("float_enabled", False)
            desired = False
        if not granted:
            if running:
                self.core.bridge.float_stop()
                running = False
            self._float_running = False
            self._mini_visible = desired
            self._apply_orb_visibility()
            return
        if desired and not running:
            outcome = self.core.bridge.float_start(
                58, self.core.config.sphere_color, self.core.config.sphere_style
            )
            if outcome.get("ok"):
                running = True
            else:
                LOG.warning(f"float start failed: {outcome.get('error')}")
        elif running and not desired:
            self.core.bridge.float_stop()
            running = False
        self._float_running = running
        self._mini_visible = desired and not running
        self._apply_orb_visibility()

    def _maybe_prompt_assets(self) -> None:
        if self._smoke_mode or getattr(self, "_asset_prompt_shown", False):
            return
        self._asset_prompt_shown = True
        try:
            folders = ensure_asset_dirs()
            root = folders.get("phi3").parent if folders.get("phi3") else ASSETS_DIR
            if not self.core.config.assets_path:
                self.core.config.set("assets_path", str(root))
            rows = build_asset_status(self.core, getattr(self, "_startup_report", None))
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"asset scan failed: {exc}")
            return
        if not rows or all(row["ok"] for row in rows):
            return
        modal = BoshraModal(self, T("Model folders", "مجلدات النماذج"), self.palette,
                            height=min(Window.height * 0.70, dp(560)))
        modal.text(T(
            "Boshra scans these folders at every start and creates them when missing. "
            "Add Phi-3, Vosk and Piper files here, or download them from the Models screen.",
            "تفحص بشرى هذه المجلدات عند كل تشغيل وتنشئها إن لم تكن موجودة. "
            "أضف ملفات Phi-3 وVosk وPiper هنا، أو نزّلها من شاشة النماذج.",
        ), 9.2)
        modal.text(T("Assets folder: ", "مجلد الأصول: ") + str(root), 8.2, "accent")
        for row in rows:
            modal.text(
                ("● " if row["ok"] else "○ ") + row["label"] + " — " + row["detail"]
                + "\n" + row["path"],
                8.0, "good" if row["ok"] else "danger",
            )
        modal.button(T("Open Models", "فتح النماذج"), lambda: self._open_models_from_modal(modal))
        modal.button(T("Later", "لاحقاً"), lambda: modal.dismiss())
        modal.open()

    def _open_models_from_modal(self, modal: Any) -> None:
        with contextlib.suppress(Exception):
            modal.dismiss()
        self.show_screen("models")

    def _ensure_welcome(self, chat: Any) -> None:
        if chat is None or getattr(chat, "log_box", None) is None:
            return
        if chat.log_box.children:
            return
        chat.add_message("assistant", self._welcome_text())

    @staticmethod
    def _welcome_text() -> str:
        english = (
            "Private and ready. Ask in Arabic or English, import documents in Vault, "
            "or open Diagnostics. Without a GGUF model I use the deterministic local "
            "engine and say so honestly — nothing leaves this device."
        )
        arabic = (
            "جاهزة وخصوصيتك محفوظة. اسأل بالعربية أو الإنجليزية، واستورد مستنداتك في الخزنة، "
            "أو افتح التشخيص. بدون نموذج GGUF أستخدم المحرك المحلي الحتمي وأقول ذلك بصراحة — "
            "لا شيء يغادر جهازك."
        )
        if _UI_LANG == "ar":
            return arabic
        if _UI_LANG == "en":
            return english
        return f"{english}\n\n{arabic}"

    # -- layout ----------------------------------------------------------
    def _refresh_insets(self) -> None:
        insets: Dict[str, int] = {}
        with contextlib.suppress(Exception):
            insets = self.core.bridge.window_insets() or {}
        previous = (getattr(self, "_safe_top", 0), getattr(self, "_safe_bottom", 0))
        self._insets = insets
        self._safe_top = min(int(insets.get("top") or 0), dp(30))
        self._safe_bottom = min(int(insets.get("bottom") or 0), dp(24))
        self._insets_changed = previous != (self._safe_top, self._safe_bottom)

    def _refresh_insets_safe(self, _dt: float = 0.0) -> None:
        self._refresh_insets()
        if getattr(self, "_insets_changed", False) and hasattr(self, "_root"):
            self.update_layout()

    def update_layout(self, *_args: Any) -> None:
        if not hasattr(self, "_root") or self._root is None:
            return
        width, height = self._root.size
        if width <= 1 or height <= 1:
            return
        safe_top = getattr(self, "_safe_top", 0)
        safe_bottom = getattr(self, "_safe_bottom", 0)
        usable_height = max(dp(200), height - safe_top - safe_bottom)
        compact = usable_height < dp(640)
        tall = usable_height >= dp(790)
        header_h = dp(50 if compact else (62 if tall else 56))
        nav_h = dp(64 if compact else (86 if tall else 76))
        telem_w = dp(94) if width < dp(360) else dp(112)
        frame_w = dp(2)
        host_top = height - safe_top - header_h
        host_height = max(dp(80), host_top - nav_h - safe_bottom)

        if getattr(self, "background", None) is not None:
            self.background.pos = (0, 0)
            self.background.size = (width, height)

        self.header.pos = (0, height - safe_top - header_h)
        self.header.size = (width, header_h)

        self.nav_grid.pos = (0, safe_bottom)
        self.nav_grid.size = (width, nav_h)

        chat_mode = self.current_screen == "chat"
        telem_height = min(dp(180), host_height * 0.27 - dp(6)) if chat_mode else min(dp(192), host_height - dp(8))
        telem_visible = bool(self.telem_enabled) and chat_mode and telem_height >= dp(148)
        if telem_visible:
            self.telemetry.pos = (dp(5), host_top - telem_height - dp(4))
            self.telemetry.size = (telem_w, telem_height)
            self.telemetry.opacity = 1
        else:
            self.telemetry.pos = (-telem_w - dp(30), host_top - dp(200))
            self.telemetry.size = (telem_w, max(dp(120), telem_height))
            self.telemetry.opacity = 0

        host_x = frame_w if (chat_mode or not telem_visible) else telem_w + dp(10)
        self.host.pos = (host_x, safe_bottom + nav_h)
        self.host.size = (max(dp(80), width - host_x - frame_w), host_height)

        if chat_mode:
            screen = self.screens.get("chat")
            band_ratio = float(getattr(screen, "sphere_band", 0.34))
            band_top = host_top - dp(4)
            band_bottom = self.host.y + host_height * (1.0 - band_ratio) + dp(4)
            band_height = max(dp(56), band_top - band_bottom)
            diameter = min(width * 0.62, band_height)
            center_x = width * 0.5
            if telem_visible:
                telem_right = self.telemetry.x + self.telemetry.width + dp(8)
                if center_x - diameter * 0.5 < telem_right:
                    room = max(dp(64), width - dp(6) - telem_right)
                    diameter = min(diameter, room)
                    center_x = telem_right + room * 0.5
            center_y = band_bottom + band_height * 0.5
        else:
            diameter = min(width * 0.55, height * 0.29)
            center_x = width * 0.5
            center_y = safe_top + (height - safe_top) * 0.80
            center_y = min(center_y, host_top - diameter * 0.5 - dp(3))
            center_y = max(center_y, safe_bottom + nav_h + diameter * 0.5 + dp(3))
        self.sphere_touch.size = (diameter, diameter)
        self.sphere_touch.pos = (center_x - diameter * 0.5, center_y - diameter * 0.5)
        if getattr(self, "background", None) is not None:
            self.background.sphere_center = [center_x / max(1.0, width),
                                             center_y / max(1.0, height)]
            self.background.sphere_radius = diameter * 0.5 / max(1.0, height)

        if self.toast_panel is not None:
            toast_width = min(width - dp(16), dp(350))
            self.toast_panel.size = (toast_width, dp(32))
            self.toast_panel.pos = ((width - toast_width) * 0.5, safe_bottom + nav_h + dp(7))
        self._place_orb()
        self._apply_orb_visibility()

    def _place_orb(self, initial: bool = False) -> None:
        orb = getattr(self, "float_orb", None)
        if orb is None or not hasattr(self, "_root"):
            return
        width, height = self._root.size
        if width <= 1 or height <= 1:
            return
        safe_bottom = getattr(self, "_safe_bottom", 0)
        nav_height = getattr(getattr(self, "nav_grid", None), "height", 0) or 0
        composer_clear = dp(86) if self.current_screen == "chat" else 0
        floor_y = safe_bottom + nav_height + composer_clear + dp(8)
        if initial:
            saved = self.core.store.state_get("float:pos", "")
            placed = False
            if "," in saved:
                with contextlib.suppress(ValueError):
                    saved_x, saved_y = (int(part) for part in saved.split(",", 1))
                    orb.pos = (saved_x, saved_y)
                    placed = True
            if not placed:
                orb.pos = (width - orb.width - dp(10), floor_y + dp(16))
        orb.x = clamp(orb.x, dp(4), max(dp(4), width - orb.width - dp(4)))
        orb.y = clamp(orb.y, floor_y, max(floor_y, height - orb.height - dp(34)))

    def _apply_orb_visibility(self) -> None:
        orb = getattr(self, "float_orb", None)
        if orb is None:
            return
        visible = bool(getattr(self, "_mini_visible", False)) and self.current_screen == "chat"
        if getattr(self, "_float_running", False):
            visible = False
        orb.opacity = 1 if visible else 0
        orb.size = (dp(54), dp(54)) if visible else (dp(1), dp(1))

    def toggle_float(self) -> None:
        config = self.core.config
        if IS_ANDROID:
            status = self.core.bridge.float_status()
            if status.get("supported"):
                if not status.get("granted"):
                    self.core.bridge.request_overlay_permission()
                    config.set("float_enabled", True)
                    self._mini_visible = False
                    self._apply_orb_visibility()
                    self.toast(T(
                        "Allow “display over other apps”, then turn Float on again.",
                        "اسمح بـ«العرض فوق التطبيقات الأخرى»، ثم فعّل العائم مجدداً.",
                    ), 6)
                    return
                if status.get("running"):
                    outcome = self.core.bridge.float_stop()
                    self._float_running = False
                    config.set("float_enabled", False)
                    self._mini_visible = False
                    self._apply_orb_visibility()
                    self.toast(str(outcome.get("message") or outcome.get("error") or ""), 3.5)
                else:
                    outcome = self.core.bridge.float_start(
                        58, config.sphere_color, config.sphere_style
                    )
                    if outcome.get("ok"):
                        self._float_running = True
                        config.set("float_enabled", True)
                        self._mini_visible = False
                        self._apply_orb_visibility()
                        self.toast(T("Floating assistant is on.", "المساعد العائم يعمل."), 3.5)
                    else:
                        self._mini_visible = True
                        self._apply_orb_visibility()
                        self.toast(str(outcome.get("error") or "Float unavailable"), 5)
                return
        enabled = not bool(config.float_enabled)
        config.set("float_enabled", enabled)
        self._mini_visible = enabled
        self._apply_orb_visibility()
        self._place_orb(initial=True)
        self.toast(T("Floating assistant ", "المساعد العائم ") + (
            T("shown", "ظاهر") if enabled else T("hidden", "مخفي")
        ), 2.4)

    def open_mini_assistant(self) -> None:
        MiniAssistantModal(self).open()

    def show_plan_modal(self, goal: str = "") -> None:
        PlanModal(self, goal).open()

    def show_recall_modal(self, query: str = "") -> None:
        RecallModal(self, query).open()

    def show_backup_modal(self) -> None:
        BackupModal(self).open()

    def show_latest_digest(self) -> None:
        digest = self.core.proactive.latest() if self.core.proactive else None
        if not digest:
            outcome = self.core.tools.execute("digest", {"kind": "manual"})
            digest = outcome.data if isinstance(outcome.data, dict) else None
        if not digest:
            self.toast(T("No brief is available yet.", "لا يوجد موجز بعد."))
            return
        DigestModal(self, str(digest.get("body", "")), str(digest.get("kind", "manual"))).open()

    def _on_keyboard(self, _window: Any, key: int, _scancode: int, _codepoint: str, _modifier: Any) -> bool:
        if key != 27:
            return False
        if self.modals:
            with contextlib.suppress(Exception):
                self.modals[-1].dismiss()
            return True
        if self.current_screen != "chat":
            self.ui_guard(lambda: self.show_screen("chat"))
            return True
        return False

    # -- screens and chrome ---------------------------------------------
    def ensure_screen(self, key: str) -> Any:
        key = SCREEN_ALIASES.get(key, "chat")
        if key in self.screens:
            return self.screens[key]
        screen_class = SCREEN_CLASSES.get(key, ChatScreen)
        try:
            screen = screen_class(self)
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"screen {key} failed to build", exc)
            _append_error_dump(f"screen-{key}", traceback.format_exc())
            fallback = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(6))
            fallback.add_widget(WrapLabel(
                text=shape_text(T(
                    "This screen could not be loaded. Details were written to the error log.",
                    "تعذر تحميل هذه الشاشة. كُتبت التفاصيل في سجل الأخطاء.",
                )),
                font_size=sp(11), color=_rgba(self.palette["danger"]),
            ))
            screen = fallback
        self.screens[key] = screen
        return screen

    def show_screen(self, key: str) -> None:
        key = SCREEN_ALIASES.get(key, "chat")
        if key not in SCREEN_CLASSES:
            key = "chat"
        self.current_screen = key
        screen = self.ensure_screen(key)
        if hasattr(self, "host"):
            self.host.clear_widgets()
            self.host.add_widget(screen)
        if key == "chat":
            self._ensure_welcome(screen)
            with contextlib.suppress(Exception):
                screen.refresh_status()
        else:
            refresh = getattr(screen, "refresh", None)
            if callable(refresh):
                with contextlib.suppress(Exception):
                    refresh()
        self.update_nav()
        self.update_layout()

    def update_nav(self) -> None:
        for key, button in self.nav_buttons.items():
            active = key == self.current_screen
            button.background_color = _rgba(self.palette["accent"], 0.36 if active else 0.11)
            button.color = _rgba(self.palette["text"], 1.0 if active else 0.72)

    def update_chrome_status(self) -> None:
        if getattr(self, "status_label", None) is None:
            return
        try:
            engine = "llama.cpp" if self.core.llm.engine_name.startswith("llama") else "heuristic"
            mode = str(self.core.config.online_mode).upper()
            self.status_label.text = f"{engine} · {mode}"
            self.status_label.color = _rgba(
                self.palette["good"] if engine == "llama.cpp" else self.palette["dim"]
            )
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"chrome status: {exc}")

    def update_telemetry(self) -> None:
        if not getattr(self, "telem_labels", None):
            return
        try:
            data = self.core.telemetry()
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"telemetry failed: {exc}")
            return
        battery = data.get("battery") or {}
        network = data.get("network") or {}
        temperature = battery.get("temperature_c")
        percent = battery.get("percent")
        level = f"{float(percent):.0f}%" if isinstance(percent, (int, float)) else "--"
        values = {
            "ram": f"{float(data.get('ram_percent') or 0):.0f}%",
            "cpu": f"{float(data.get('cpu') or 0):.0f}%",
            "bat": f"{level}{'+' if battery.get('charging') else ''}",
            "temp": (f"{float(temperature):.0f}C" if isinstance(temperature, (int, float)) else "--"),
            "free": human_bytes(int(data.get("storage_free") or 0)),
            "net": str(network.get("type") or ("on" if network.get("online") else "off"))[:9],
            "cores": str(data.get("cores") or "--"),
            "uptime": f"{int(data.get('uptime') or 0) // 60}m",
        }
        for key, value in values.items():
            label = self.telem_labels.get(key)
            if label is not None:
                label.text = f"{key.upper():<5} {value}"

    def set_state(self, state: str) -> None:
        state = state if state in STATE_VALUES else "idle"
        if getattr(self, "background", None) is not None:
            self.background.target_state = STATE_VALUES[state]
            self.background.energy = STATE_ENERGY[state]
        if getattr(self, "state_label", None) is not None:
            english, arabic = STATE_LABELS[state]
            self.state_label.text = shape_text(T(english, arabic))
            color = {
                "idle": "accent", "thinking": "accent2",
                "speaking": "danger", "listening": "good",
            }.get(state, "accent")
            self.state_label.color = _rgba(self.palette[color])

    def toggle_telemetry(self) -> None:
        self.telem_enabled = not self.telem_enabled
        self.update_layout()
        self.toast(T("Telemetry panel ", "لوحة القياس ") + (
            T("shown", "ظاهرة") if self.telem_enabled else T("hidden", "مخفية")
        ), 1.6)

    # -- events ----------------------------------------------------------
    def _ui_event(self, event: str, payload: Dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            Clock.schedule_once(lambda _dt: self.handle_event(event, payload), 0)

    def _ui_after(self, callback: Callable[..., Any], *args: Any) -> None:
        with contextlib.suppress(Exception):
            Clock.schedule_once(lambda _dt: self.ui_guard(lambda: callback(*args)), 0)

    def handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self._shutting_down:
            return
        try:
            if event == "state":
                self.set_state(str(payload.get("state", "idle")))
            elif event == "toast":
                self.toast(str(payload.get("text", "")))
            elif event == "navigate":
                self.show_screen(str(payload.get("screen", "chat")))
            elif event == "theme":
                self.apply_theme(str(payload.get("theme", self.core.config.theme)))
            elif event == "profile":
                self.core.config.set("profile", str(payload.get("profile", self.core.config.profile)))
                self.toast(T("Hardware profile updated", "تم تحديث ملف الموارد"))
            elif event == "focus":
                minutes = payload.get("minutes", 25)
                self.toast(T(f"Focus session started: {minutes} minutes", f"بدأت جلسة التركيز: {minutes} دقيقة"), 3.5)
                self.core.bridge.vibrate(30)
            elif event == "workflow":
                flow = self.screens.get("flow")
                handler = getattr(flow, "on_workflow", None)
                if callable(handler):
                    handler(payload)
            elif event == "tts":
                state = str(payload.get("state", "idle"))
                self._tts_speaking = state == "speaking"
                if state == "speaking":
                    self.set_state("speaking")
                elif not self.core.busy:
                    self.set_state("idle")
            elif event == "voice_state":
                state = str(payload.get("state", "idle"))
                if state in ("listening", "wake"):
                    if not self.core.busy:
                        self.set_state("listening")
                elif not self.core.busy and not self._tts_speaking:
                    self.set_state("idle")
            elif event == "voice_partial":
                chat = self.screens.get("chat")
                if chat is not None and hasattr(chat, "set_partial"):
                    chat.set_partial(str(payload.get("text", "")))
            elif event == "voice_final":
                self.send_voice_text(str(payload.get("text", "")), str(payload.get("lang", "en")))
            elif event == "voice_error":
                self.toast(str(payload.get("text", "")), 5)
                if not self.core.busy:
                    self.set_state("idle")
            elif event == "proactive":
                digest = payload.get("digest") or {}
                body = str(digest.get("body", ""))
                if body:
                    self.toast(T("New brief ready · ", "موجز جديد جاهز · ") + body.split("\n", 2)[-1][:90], 6)
            elif event == "plan":
                self.show_screen("chat")
            elif event == "float":
                self._sync_float()
                self.toast(str(payload.get("text", "")), 3.5)
            elif event == "reset":
                self.rebuild_ui()
        except Exception as exc:  # noqa: BLE001
            LOG.error(f"event {event} failed", exc)
            _append_error_dump(f"event-{event}", traceback.format_exc())

    # -- chat ------------------------------------------------------------
    def send_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self._shutting_down:
            return
        if self.core.busy or self.core.orchestrator._processing:
            self.toast(T("Boshra is still answering. Say stop to interrupt.", "بشرى ما زالت تجيب. قل توقف للمقاطعة."))
            return
        chat = self.ensure_screen("chat")
        if self.current_screen != "chat":
            self.show_screen("chat")
        try:
            self._ensure_welcome(chat)
            chat.add_message("user", text)
            chat.begin_assistant()
        except Exception as exc:  # noqa: BLE001
            LOG.error("chat append failed", exc)
            _append_error_dump("chat-append", traceback.format_exc())
        self._dispatch_reply(text, chat, record_user=True)

    def _dispatch_reply(self, text: str, chat: Any, record_user: bool = True) -> None:
        self.core.busy = True
        self.set_state("thinking")

        def on_token(token: str) -> None:
            self._ui_after(chat.append_token, token)

        def on_state(state: str) -> None:
            self._ui_after(self.set_state, state)

        def on_done(final_text: str, meta: Dict[str, Any]) -> None:
            self._ui_after(self._finish_reply, chat, final_text or "", meta or {})

        def worker() -> None:
            try:
                self.core.orchestrator.process(
                    text, on_token=on_token, on_state=on_state, on_done=on_done,
                    record_user=record_user,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.error("chat worker failed", exc)
                _append_error_dump("chat-worker", traceback.format_exc())
                on_done(T("A local error interrupted the reply.", "قاطع خطأ محلي الرد."), {"intent": "error"})

        threading.Thread(target=worker, name="boshra-chat", daemon=True).start()

    def regenerate_last(self) -> None:
        """Re-run the latest user turn and replace the visible reply."""
        if self.core.busy or self.core.orchestrator._processing:
            self.toast(T("Wait for the current reply or stop it first.", "انتظر الرد الحالي أو أوقفه أولاً."))
            return
        history = self.core.orchestrator.history(limit=200)
        last_user = next((item for item in reversed(history) if item["role"] == "user"), None)
        if last_user is None:
            self.toast(T("There is no message to regenerate yet.", "لا توجد رسالة لإعادة توليدها بعد."))
            return
        chat = self.ensure_screen("chat")
        chat.remove_last_assistant()
        self.core.orchestrator.drop_trailing_assistants()
        chat.begin_assistant()
        if self.current_screen != "chat":
            self.show_screen("chat")
        self._dispatch_reply(str(last_user["content"]), chat, record_user=False)
        self.toast(T("Regenerating the last reply.", "أعيد توليد الرد الأخير."), 1.6)

    def speak_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not self.core.tts.available:
            self.toast(T("Speech output is unavailable on this device.", "الإخراج الصوتي غير متاح على هذا الجهاز."))
            return
        if not self.core.config.tts_enabled:
            self.toast(T("Speech output is disabled in Settings.", "الإخراج الصوتي معطّل في الإعدادات."))
            return
        with contextlib.suppress(Exception):
            self.core.tts.stop()
        self.core.tts.speak(text)

    def edit_in_composer(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        chat = self.ensure_screen("chat")
        if self.current_screen != "chat":
            self.show_screen("chat")
        chat.input.text = text
        with contextlib.suppress(Exception):
            chat.input.cursor = (len(text), 0)
            chat.input.focus = True
        self.toast(T("Loaded into the composer for editing.", "حُمّل في المحرر للتعديل."), 1.6)

    def send_voice_text(self, text: str, _lang: str = "en") -> None:
        text = (text or "").strip()
        if not text:
            return
        chat = self.screens.get("chat")
        if chat is not None and hasattr(chat, "set_partial"):
            with contextlib.suppress(Exception):
                chat.set_partial("")
        self.send_text(text)

    def _finish_reply(self, chat: Any, final_text: str, meta: Dict[str, Any]) -> None:
        self.core.busy = False
        if chat is not None and hasattr(chat, "finish_assistant"):
            with contextlib.suppress(Exception):
                chat.finish_assistant(final_text)
        if not self._tts_speaking:
            self.set_state("idle")
        self.update_chrome_status()
        if chat is not None and hasattr(chat, "refresh_status"):
            with contextlib.suppress(Exception):
                chat.refresh_status()
        if meta.get("intent") == "error":
            self.toast(T("A local error interrupted processing. See the error log.", "قاطع خطأ محلي المعالجة. راجع سجل الأخطاء."), 5)

    def new_conversation(self) -> None:
        with contextlib.suppress(Exception):
            if self.core.busy:
                self.toast(T("Wait for the current reply or stop it first.", "انتظر الرد الحالي أو أوقفه أولاً."))
                return
            self.core.orchestrator.new_conversation()
            chat = self.ensure_screen("chat")
            if hasattr(chat, "log_box"):
                chat.log_box.clear_widgets()
            if hasattr(chat, "reset_timeline"):
                chat.reset_timeline()
            if hasattr(chat, "input"):
                chat.input.text = ""
            self.show_screen("chat")
            self._ensure_welcome(chat)
            if hasattr(chat, "_on_composer_changed"):
                chat._on_composer_changed()
            self.toast(T("New conversation", "محادثة جديدة"), 1.8)

    def stop_all(self) -> None:
        with contextlib.suppress(Exception):
            self.core.orchestrator.stop_event.set()
        with contextlib.suppress(Exception):
            self.core.tts.stop()
        with contextlib.suppress(Exception):
            self.core.automation.cancel()
        voice = self.core.voice
        if voice is not None and getattr(voice, "state", "idle") != "idle":
            self.run_bg(voice.stop, lambda _result: None)
        self._tts_speaking = False
        self.set_state("idle")
        self.toast(T("Stopped.", "تم الإيقاف."), 1.5)

    # -- voice -----------------------------------------------------------
    def toggle_voice(self, mode: str = "command") -> None:
        voice = self.core.voice
        if voice is None:
            self.toast(T("Voice input is unavailable.", "الإدخال الصوتي غير متاح."), 4)
            return
        if getattr(voice, "state", "idle") != "idle":
            self.run_bg(voice.stop, lambda _result: None)
            self.set_state("idle")
            self.toast(T("Listening stopped.", "تم إيقاف الاستماع."), 1.6)
            return
        if not voice.available:
            self.toast(T("Voice unavailable: ", "الصوت غير متاح: ") + voice.status, 6)
            return

        def work() -> Dict[str, Any]:
            allowed, reason = self.core.permissions.require(
                ["microphone.listen"],
                purpose="Foreground voice input",
                risk="medium",
                detail="Microphone capture runs only while the microphone control is active.",
            )
            if not allowed:
                return {"ok": False, "message": reason}
            started = voice.start(mode)
            return {"ok": bool(started), "message": voice.status, "mode": mode}

        def done(result: Any) -> None:
            result = result or {}
            if result.get("ok"):
                self.set_state("listening")
                self.toast(T("Listening. Speak now.", "أستمع. تحدث الآن."), 2.0)
                self.core.bridge.vibrate(20)
            else:
                self.set_state("idle")
                self.toast(T("Voice unavailable: ", "الصوت غير متاح: ") + str(result.get("message", "")), 6)

        self.run_bg(work, done)

    def _start_accelerometer(self) -> None:
        try:
            from plyer import accelerometer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"accelerometer import failed: {exc}")
            return
        try:
            accelerometer.enable()
            self._accel = accelerometer
            self._accel_event = Clock.schedule_interval(self._read_accelerometer, 0.12)
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"accelerometer unavailable: {exc}")
            self._accel = None

    def _read_accelerometer(self, _dt: float) -> None:
        if self._accel is None or not self.core.config.animations:
            return
        try:
            values = self._accel.acceleration
            if not values or len(values) < 3:
                return
            x, y, _z = (float(values[0]), float(values[1]), float(values[2]))
            self.background.target_tilt = [
                clamp(y / 9.81 * 0.7, -0.85, 0.85),
                clamp(x / 9.81 * 0.7, -0.85, 0.85),
            ]
        except Exception:
            self._accel = None
            if self._accel_event is not None:
                with contextlib.suppress(Exception):
                    self._accel_event.cancel()
                self._accel_event = None

    # -- periodic --------------------------------------------------------
    def tick(self, _dt: float) -> None:
        if self._shutting_down:
            return
        self._tick_counter += 1
        now = datetime.now()
        with contextlib.suppress(Exception):
            time_text = now.strftime("%H:%M")
            date_text = hijri_date(now) if self.core.config.hijri_dates else now.strftime("%a %d %b")
            if self.core.config.arabic_numerals:
                time_text = localize_digits(time_text)
                date_text = localize_digits(date_text)
            self.clock_label.text = time_text
            self.date_label.text = date_text
        if self._accel is None and getattr(self, "background", None) is not None and self.core.config.animations:
            self.background.target_tilt = [
                math.sin(time.time() * 0.12) * 0.22,
                math.cos(time.time() * 0.09) * 0.18,
            ]
        if self._tick_counter % 2 == 0:
            self.update_telemetry()
        if self._tick_counter % 6 == 0:
            self.update_chrome_status()
        if IS_ANDROID and self._tick_counter % 20 == 0:
            self._sync_float()
        self._check_reminders()
        self._check_focus()
        if self._tick_counter % 4 == 0:
            self._check_proactive()
        if self._tick_counter % 120 == 0:
            self._maintenance()

    def _check_proactive(self) -> None:
        if not self.core.config.proactive_enabled or self.core.proactive is None:
            return
        with contextlib.suppress(Exception):
            self.core.proactive.check()

    def _check_reminders(self) -> None:
        now = time.time()
        if now - self._last_reminder_scan < 2.0:
            return
        self._last_reminder_scan = now
        stamp = datetime.now().isoformat(timespec="seconds")
        rows = self.core.store.query(
            "SELECT * FROM reminders WHERE fired=0 AND at<=? ORDER BY at LIMIT 5", (stamp,)
        )
        for row in rows:
            self.core.store.execute("UPDATE reminders SET fired=1 WHERE id=?", (row["id"],))
            self.core.store.audit("reminder", "fired", str(row.get("title", ""))[:200], "low")
            title = str(row.get("title", ""))
            self.toast(T("Reminder: ", "تذكير: ") + title, 6)
            with contextlib.suppress(Exception):
                self.core.bridge.notify(APP_NAME, title)
            self.core.bridge.vibrate(60)
            if self.core.config.tts_enabled:
                with contextlib.suppress(Exception):
                    self.core.tts.speak(T("Reminder: ", "تذكير: ") + title, detect_language(title))

    def _check_focus(self) -> None:
        end = float(self.core.ctx.focus_end or 0.0)
        if not end or time.time() < end:
            return
        self.core.ctx.focus_end = 0.0
        with contextlib.suppress(Exception):
            self.core.store.state_set("focus_end", "0")
        self.toast(T("Focus session complete.", "اكتملت جلسة التركيز."), 5)
        self.core.bridge.vibrate(120)
        if self.core.config.tts_enabled:
            with contextlib.suppress(Exception):
                self.core.tts.speak(
                    T("Focus session complete.", "اكتملت جلسة التركيز."),
                    _UI_LANG if _UI_LANG in ("ar", "en") else "en",
                )

    def _maintenance(self) -> None:
        now = time.time()
        if now - self._last_maintenance < 60:
            return
        self._last_maintenance = now
        try:
            if self.core.llm.engine_name.startswith("llama") and self.core.resources.should_unload_model():
                self.core.llm.unload()
                self.toast(T("Model unloaded under memory or thermal pressure.", "تم تفريغ النموذج لتخفيف ضغط الذاكرة أو الحرارة."), 5)
                self.update_chrome_status()
            retention = int(self.core.config.retention_days or 0)
            if retention > 0:
                purged = self.core.memory.purge_expired(retention)
                if purged:
                    LOG.info(f"retention purged {purged} memories")
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"maintenance failed: {exc}")

    # -- theme, modal, helpers ------------------------------------------
    def apply_theme(self, name: str) -> None:
        name = name if name in PALETTES else "aurora"
        self.core.config.set("theme", name)
        self.palette = PALETTES[name]
        self._post_rebuild_toast = T("Theme: ", "المظهر: ") + name
        self.rebuild_ui()

    def rebuild_ui(self) -> None:
        if self._shutting_down or self._rebuild_scheduled:
            return
        self._rebuild_scheduled = True
        Clock.schedule_once(self._do_rebuild, 0.01)

    def _do_rebuild(self, _dt: float) -> None:
        self._rebuild_scheduled = False
        if self._shutting_down:
            return
        self.palette = PALETTES.get(self.core.config.theme, PALETTES["aurora"])
        global BUTTON_STYLE
        BUTTON_STYLE = self.core.config.button_style
        self._compose()

    def confirm_sync(self, question: str, detail: str = "") -> bool:
        if self._shutting_down:
            return False
        if threading.current_thread() is threading.main_thread():
            LOG.warning("blocking confirmation requested on the UI thread; denying to avoid a deadlock")
            self.toast(T("This action must run in the background.", "يجب تنفيذ هذا الإجراء في الخلفية."))
            return False
        completed = threading.Event()
        answer = {"value": False}

        def finish(value: bool) -> None:
            answer["value"] = bool(value)
            completed.set()

        def show(_dt: float) -> None:
            try:
                ConfirmModal(self, question, detail, self.palette, finish).open()
            except Exception as exc:  # noqa: BLE001
                LOG.error("confirmation modal failed", exc)
                completed.set()

        with contextlib.suppress(Exception):
            Clock.schedule_once(show, 0)
        if not completed.wait(timeout=180):
            LOG.warning("confirmation timed out")
            return False
        return bool(answer["value"])

    def confirm_async(self, question: str, detail: str, callback: Callable[[bool], Any]) -> None:
        def finish(value: bool) -> None:
            self.ui_guard(lambda: callback(bool(value)))
        try:
            ConfirmModal(self, question, detail, self.palette, finish).open()
        except Exception as exc:  # noqa: BLE001
            LOG.error("confirmation dialog failed", exc)
            _append_error_dump("confirm-dialog", traceback.format_exc())
            finish(False)

    def pick_path(self) -> Optional[str]:
        if IS_ANDROID:
            try:
                return self.core.bridge.pick_file()
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"android file picker failed: {exc}")
                return None
        for executable, args in (
            ("zenity", ["--file-selection", "--title=Select a file for Boshra"]),
            ("kdialog", ["--getopenfilename", str(Path.home())]),
        ):
            if shutil.which(executable):
                try:
                    completed = subprocess.run(
                        [executable, *args], capture_output=True, text=True, timeout=120
                    )
                    selected = (completed.stdout or "").strip()
                    return selected or None
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(f"{executable} picker failed: {exc}")
        try:
            return self.core.bridge.pick_file()
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"desktop file picker failed: {exc}")
            return None

    def toast(self, text: str, duration: float = 2.8) -> None:
        if not text or self.toast_label is None:
            return
        self.toast_label.text = shape_text(str(text)[:320])
        self.toast_panel.opacity = 1
        if self._toast_event is not None:
            with contextlib.suppress(Exception):
                self._toast_event.cancel()
        self._toast_event = Clock.schedule_once(self._hide_toast, max(0.8, float(duration)))

    def _hide_toast(self, _dt: float) -> None:
        if self.toast_panel is not None:
            self.toast_panel.opacity = 0
        self._toast_event = None

    def trigger_shock(self, x: float, y: float) -> None:
        if getattr(self, "background", None) is None:
            return
        width, height = max(1.0, self._root.width), max(1.0, self._root.height)
        self.background.shock_origin = [clamp(x / width, 0.0, 1.0), clamp(y / height, 0.0, 1.0)]
        self.background.shock_time = Clock.get_boottime()
        with contextlib.suppress(Exception):
            self.core.bridge.vibrate(28)
        with contextlib.suppress(Exception):
            self.core.store.audit("ui", "sphere-touch", "", "low")

    def ui_guard(self, callback: Callable[[], Any]) -> Any:
        try:
            return callback()
        except Exception as exc:  # noqa: BLE001
            LOG.error("UI action failed", exc)
            _append_error_dump("ui-action", traceback.format_exc())
            with contextlib.suppress(Exception):
                self.toast(T("Action failed. Details were written to the error log.", "فشل الإجراء. كُتبت التفاصيل في سجل الأخطاء."), 5)
            return None

    def run_bg(self, function: Callable[[], Any], callback: Optional[Callable[[Any], Any]] = None) -> None:
        def worker() -> None:
            try:
                result = function()
            except Exception as exc:  # noqa: BLE001
                LOG.error("background task failed", exc)
                _append_error_dump("background-task", traceback.format_exc())
                self._ui_after(self.toast, T("Background task failed. See the error log.", "فشلت مهمة خلفية. راجع سجل الأخطاء."))
                return
            if callback is not None:
                self._ui_after(callback, result)

        threading.Thread(target=worker, name="boshra-bg", daemon=True).start()

    # -- lifecycle -------------------------------------------------------
    def on_pause(self) -> bool:
        with contextlib.suppress(Exception):
            if self.core.voice is not None:
                self.core.voice.stop()
        with contextlib.suppress(Exception):
            self.core.tts.stop()
        return True

    def on_resume(self) -> bool:
        self._refresh_insets_safe(0)
        self.update_chrome_status()
        self.update_telemetry()
        self._sync_float()
        self._check_proactive()
        action = ""
        with contextlib.suppress(Exception):
            action = self.core.bridge.launch_action()
        if action in ("float_chat", "chat", "assistant"):
            self.show_screen("chat")
            chat = self.screens.get("chat")
            if action == "float_chat":
                Clock.schedule_once(lambda _dt: self.open_mini_assistant(), 0.25)
            elif chat is not None:
                with contextlib.suppress(Exception):
                    chat.restore_draft(focus=True)
        return True

    def _smoke_capture(self, _dt: float) -> None:
        destination = os.environ.get("BOSHRA_SMOKE_SHOT", "/tmp/boshra_smoke.png")
        try:
            Window.screenshot(name=destination)
            print(f"[SMOKE] screenshot -> {destination}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[SMOKE] screenshot failed: {exc}", flush=True)
        try:
            print(
                f"[SMOKE] engine={self.core.llm.engine_name} skills={len(self.core.skills.list_skills())} "
                f"encryption={self.core.store.crypto.available} tts={self.core.tts.available} "
                f"voice={self.core.voice.available if self.core.voice else False}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[SMOKE] status failed: {exc}", flush=True)
        Clock.schedule_once(lambda _dt2: self.stop(), 0.25)

    def on_stop(self) -> None:
        self._shutting_down = True
        self._cancel_timers()
        with contextlib.suppress(Exception):
            Window.unbind(on_keyboard=self._on_keyboard)
        with contextlib.suppress(Exception):
            self.core.shutdown()


# ==========================================================================
# Self-test (no window, no network, no device required)
# ==========================================================================
def run_self_tests() -> int:
    print(f"{APP_NAME} {APP_VERSION} self-test")
    print(f"private data: {PRIVATE_DIR}")
    failures = 0
    total = 0
    core: Optional[BoshraCore] = None
    temporary: List[Path] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures, total
        total += 1
        if not ok:
            failures += 1
            print(f"FAIL {name}" + (f" :: {detail}" if detail else ""))
        else:
            print(f"PASS {name}")

    try:
        core = BoshraCore()
        ok, detail = core.store.integrity()
        record("sqlite-integrity", ok, detail)
        record("config-valid", core.config.profile in VALID_PROFILES and core.config.theme in PALETTES)

        marker = "selftest_" + uuid.uuid4().hex[:10]
        memory = core.memory.add("semantic", f"self test marker {marker}", key=marker, source="self-test")
        record("memory-add", bool(memory.get("ok")), str(memory.get("error", "")))
        found = core.memory.search(marker, k=5) if memory.get("ok") else []
        record("memory-search", any(marker in item.get("content", "") for item in found))
        if memory.get("id"):
            core.memory.delete(int(memory["id"]))

        document_path = Path(tempfile.gettempdir()) / f"boshra_{marker}.txt"
        document_path.write_text(
            "The nearest clinic opens at eight in the morning and closes at six in the evening. "
            "The local emergency telephone number is one one two and it works without internet.",
            encoding="utf-8",
        )
        temporary.append(document_path)
        imported = core.vault.import_file(document_path, title="self test document")
        record("vault-import", bool(imported.get("ok")), str(imported.get("error", "")))
        document_id = imported.get("doc_id")
        if document_id:
            hits = core.vault.search("clinic hours", k=3, doc_id=int(document_id))
            record("vault-search", bool(hits))
            summary = core.vault.summarize(int(document_id))
            record("vault-summarize", bool(summary.get("ok")), str(summary.get("error", "")))
            record("vault-delete", bool(core.vault.delete_document(int(document_id))))

        for tool_name, tool_args in (
            ("time_date", {}),
            ("math", {"expression": "2+2*3"}),
            ("translate", {"text": "good morning", "target": "ar"}),
        ):
            try:
                result = core.tools.execute(tool_name, tool_args)
                record(f"tool:{tool_name}", result.ok, result.error or "")
            except Exception as exc:  # noqa: BLE001
                record(f"tool:{tool_name}", False, str(exc))

        skills = core.skills.list_skills()
        record("skills-count", len(skills) >= 30, str(len(skills)))
        check = core.skills.self_test("boshra.task_manager")
        record("skill-selftest", bool(check.get("ok")), str(check.get("error", "")))
        executed = core.skills.execute("boshra.task_manager", "list")
        record("skill-execute", bool(executed.success), executed.error or "")

        routine_name = f"self test routine {marker}"
        saved = core.automation.save_routine(routine_name, [{"tool": "time_date", "args": {}}])
        record("routine-save", bool(saved.get("ok")), str(saved.get("error", "")))
        if saved.get("ok"):
            outcome = core.automation.run_by_name(routine_name)
            record("routine-run", bool(outcome.get("ok")), str(outcome.get("error") or outcome.get("reason", "")))
            row = core.store.query_one("SELECT id FROM routines WHERE name=? ORDER BY id DESC LIMIT 1", (routine_name,))
            if row:
                core.automation.delete_routine(int(row["id"]))

        route = core.orchestrator._route("battery status")
        record("intent-route", bool(route and route[0] == "tool" and route[1].get("tool") == "battery"))
        fallback = core.llm.fallback.answer("2+2", "en")
        record("fallback-engine", "4" in str(fallback))
        record("arabic-shaping", bool(shape_text("مرحبا بكم")))
        emergency = core.emergency.search("cpr")
        record("emergency-guide", bool(emergency.get("text")))
        record("diagnostics", bool(core.diagnostics.run_all()))
        record("tts-status", bool(core.tts.status))
        asset_report = inspect_assets()
        record(
            "asset-report",
            len(asset_report.statuses) == len(_ASSET_SPECS)
            and isinstance(asset_report.as_dict().get("assets"), list),
            f"missing={','.join(asset_report.missing) or 'none'}",
        )
        empty_report = inspect_assets(PRIVATE_DIR / "missing-assets-probe")
        record(
            "asset-report-missing",
            not empty_report.ok and len(empty_report.statuses) == len(_ASSET_SPECS),
        )
        sound_ready = core.ui_sound.warm_up()
        record("ui-sound-generate", sound_ready, core.ui_sound.last_error)
        try:
            with wave.open(str(core.ui_sound.path_for("tap_glass")), "rb") as handle:
                frames = handle.getnframes()
                rate_ok = handle.getframerate() == UISound.SAMPLE_RATE
            record("ui-sound-format", frames > 500 and rate_ok, f"{frames} frames")
        except (OSError, wave.Error) as exc:
            record("ui-sound-format", False, str(exc))
        record("ui-sound-config", 0.0 <= float(core.config.ui_sound_volume) <= 1.0,
               f"volume={core.config.ui_sound_volume}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        total += 1
        print(f"FAIL self-test-crashed :: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        for path in temporary:
            with contextlib.suppress(OSError):
                path.unlink()
        if core is not None:
            with contextlib.suppress(Exception):
                core.shutdown()

    print(f"self-test: {total - failures}/{total} passed")
    return 0 if failures == 0 else 1


# ==========================================================================
# Entry point
# ==========================================================================
USAGE = """Boshra AI {version}
Usage:
  python3 main.py                     launch the application
  python3 main.py --self-test         run offline diagnostics and exit
  python3 main.py --run-skill DIR CMD [JSON]
                                      execute an external skill entry point
  python3 main.py --version
  python3 main.py --help
""".format(version=APP_VERSION)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        sys.stdout.write(USAGE)
        return 0
    if "--version" in arguments:
        print(f"{APP_NAME} {APP_VERSION} ({BUILD_STAMP})")
        return 0
    if "--self-test" in arguments:
        return run_self_tests()
    if "--run-skill" in arguments:
        index = arguments.index("--run-skill")
        remainder = arguments[index + 1:]
        if not remainder:
            sys.stderr.write("--run-skill requires a skill directory\n")
            return 2
        directory = remainder[0]
        command = remainder[1] if len(remainder) > 1 else ""
        data_json = remainder[2] if len(remainder) > 2 else "{}"
        return run_skill_file_cli(directory, command, data_json)
    apply_pending_restore()
    BoshraApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

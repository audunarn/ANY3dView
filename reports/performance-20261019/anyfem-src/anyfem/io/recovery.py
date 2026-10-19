"""Project locking and crash-recovery bundles.

This module deliberately has no Tk dependencies.  The desktop application can
use the decision records to offer read-only/open-copy choices, while headless
clients can use the same locking and autosave rules.

Locks and recovery records are data, not authority: a lock is never removed
merely because it exists, and an autosave never overwrites the user's project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from typing import Any, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

from platformdirs import user_state_path

__all__ = [
    "LockDecision",
    "LockOwner",
    "ProjectLock",
    "RecoveryCandidate",
    "RecoveryError",
    "default_recovery_root",
    "discard_recovery",
    "discover_recoveries",
    "load_recovery",
    "prune_recoveries",
    "write_autosave",
]


LOCK_SCHEMA = "anyfem.project-lock"
LOCK_VERSION = 1
RECOVERY_SCHEMA = "anyfem.recovery"
RECOVERY_VERSION = 1
_SNAPSHOT_NAME = "project.anyfem"
_METADATA_NAME = "metadata.json"
_SOURCE_UNSET = object()


class RecoveryError(ValueError):
    """Raised when a recovery bundle is malformed or unsafe."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime) -> str:
    resolved = value.astimezone(timezone.utc)
    return resolved.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _short_key(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _safe_child(root: Path, *parts: str) -> Path:
    base = root.resolve(strict=False)
    target = base.joinpath(*parts).resolve(strict=False)
    try:
        target.relative_to(base)
    except ValueError:
        raise RecoveryError("path escapes the recovery root") from None
    return target


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# ---------------------------------------------------------------------------
# Per-project lock
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LockOwner:
    pid: int
    hostname: str
    acquired_utc: str
    token: str
    process_start: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LOCK_SCHEMA,
            "version": LOCK_VERSION,
            "pid": int(self.pid),
            "hostname": self.hostname,
            "acquired_utc": self.acquired_utc,
            "token": self.token,
            "process_start": self.process_start,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LockOwner":
        if data.get("schema") != LOCK_SCHEMA:
            raise ValueError("not an ANYfem project lock")
        if int(data.get("version", 0)) != LOCK_VERSION:
            raise ValueError("unsupported project-lock version")
        pid = int(data["pid"])
        if pid <= 0:
            raise ValueError("lock PID must be positive")
        acquired = str(data["acquired_utc"])
        _parse_utc(acquired)
        hostname = str(data["hostname"])
        token = str(data["token"])
        if not hostname or not token:
            raise ValueError("lock owner is incomplete")
        marker = data.get("process_start")
        return cls(
            pid=pid,
            hostname=hostname,
            acquired_utc=acquired,
            token=token,
            process_start=None if marker in (None, "") else str(marker),
        )


LockState = Literal["available", "acquired", "owned", "live", "stale"]


@dataclass(frozen=True)
class LockDecision:
    state: LockState
    lock_path: Path
    read_only: bool
    can_take_over: bool
    owner: LockOwner | None = None
    reason: str = ""

    @property
    def writable(self) -> bool:
        return self.state in ("available", "acquired", "owned") and not self.read_only

    @property
    def acquired(self) -> bool:
        return self.state in ("acquired", "owned")


class ProjectLock:
    """Atomic-create lock beside one project.

    A stale or malformed lock is reported but not removed until
    ``acquire(take_over_stale=True)`` is explicitly requested.  Locks owned by
    another host are conservatively treated as live because their PID cannot be
    checked safely.
    """

    def __init__(self, project_path: str | Path) -> None:
        project = _resolved(project_path)
        if not project.suffix:
            project = project.with_suffix(".anyfem")
        self.project_path = project
        self.path = project.with_name(project.name + ".lock")
        self.token = str(uuid4())
        self.owner = LockOwner(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            acquired_utc=_utc_text(_utc_now()),
            token=self.token,
            process_start=_process_start_token(os.getpid()),
        )
        self._held = False

    def inspect(self) -> LockDecision:
        if not self.path.exists():
            return LockDecision("available", self.path, False, False)
        try:
            raw = self.path.read_text(encoding="utf-8")
            owner = LockOwner.from_dict(json.loads(raw))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            return LockDecision(
                "stale",
                self.path,
                True,
                True,
                reason=f"lock metadata is invalid: {error}",
            )
        if owner.token == self.token:
            return LockDecision("owned", self.path, False, False, owner)
        if owner.hostname.casefold() != socket.gethostname().casefold():
            return LockDecision(
                "live",
                self.path,
                True,
                False,
                owner,
                "the project is locked on another host; open read-only",
            )
        live = _pid_is_live(owner.pid)
        if live is False:
            return LockDecision(
                "stale", self.path, True, True, owner, "the owning process no longer exists"
            )
        current_start = _process_start_token(owner.pid) if live else None
        if (
            live
            and owner.process_start is not None
            and current_start is not None
            and owner.process_start != current_start
        ):
            return LockDecision(
                "stale", self.path, True, True, owner, "the lock PID has been reused"
            )
        reason = (
            "the project is open in another process; open read-only"
            if live
            else "the lock owner cannot be verified; open read-only"
        )
        return LockDecision("live", self.path, True, False, owner, reason)

    def acquire(self, *, take_over_stale: bool = False) -> LockDecision:
        decision = self.inspect()
        if decision.state == "owned":
            self._held = True
            return decision
        if decision.state == "stale":
            if not take_over_stale:
                return decision
            self._quarantine_stale_lock()
        elif decision.state != "available":
            return decision

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _json_bytes(self.owner.to_dict())
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            return self.inspect()
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise
        self._held = True
        return LockDecision("acquired", self.path, False, False, self.owner)

    def release(self) -> bool:
        """Remove only the lock carrying this instance's unguessable token."""

        try:
            owner = LockOwner.from_dict(
                json.loads(self.path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            self._held = False
            return False
        if owner.token != self.token:
            self._held = False
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._held = False
        return True

    def _quarantine_stale_lock(self) -> None:
        # Renaming, rather than unlinking first, keeps takeover atomic with
        # respect to other contenders.  O_EXCL below decides who actually wins.
        quarantine = self.path.with_name(
            f".{self.path.name}.{uuid4().hex}.stale"
        )
        try:
            self.path.rename(quarantine)
        except FileNotFoundError:
            return
        try:
            quarantine.unlink()
        except OSError:
            pass

    def __enter__(self) -> "ProjectLock":
        decision = self.acquire()
        if not decision.acquired:
            raise PermissionError(decision.reason or "project is locked")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


def _pid_is_live(pid: int) -> bool | None:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            process = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                0x1000, False, int(pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            )
            if not process:
                error = ctypes.get_last_error()
                # Access denied proves that something owns the PID.
                return True if error == 5 else False
            code = wintypes.DWORD()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                process, ctypes.byref(code)
            )
            ctypes.windll.kernel32.CloseHandle(process)  # type: ignore[attr-defined]
            return bool(ok and code.value == 259)  # STILL_ACTIVE
        except Exception:  # pragma: no cover - defensive platform fallback
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        return None
    return True


def _process_start_token(pid: int) -> str | None:
    """Best-effort PID-reuse marker; absence never makes a lock stale."""

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            process = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                0x1000, False, int(pid)
            )
            if not process:
                return None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            ok = ctypes.windll.kernel32.GetProcessTimes(  # type: ignore[attr-defined]
                process,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            ctypes.windll.kernel32.CloseHandle(process)  # type: ignore[attr-defined]
            if not ok:
                return None
            value = (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime)
            return f"windows-filetime:{value}"
        except Exception:  # pragma: no cover - defensive platform fallback
            return None
    stat = Path(f"/proc/{int(pid)}/stat")
    try:
        # Field 22 is process start time.  The command name may contain spaces,
        # so split only after its closing parenthesis.
        tail = stat.read_text(encoding="ascii").rsplit(")", 1)[1].split()
        return f"proc-start:{tail[19]}"
    except (OSError, IndexError, UnicodeError):
        return None


# ---------------------------------------------------------------------------
# Recovery bundles
# ---------------------------------------------------------------------------
RecoveryState = Literal[
    "unsaved", "source_unchanged", "source_changed", "source_missing", "source_unavailable"
]


@dataclass(frozen=True)
class RecoveryCandidate:
    bundle_path: Path
    snapshot_path: Path
    document_id: str
    revision_id: str
    revision_sequence: int
    document_hash: str
    model_hash: str
    saved_document_hash: str
    created_utc: str
    source_path: Path | None
    source_sha256: str | None
    source_size: int | None
    source_mtime_ns: int | None
    snapshot_sha256: str
    artifact_refs: tuple[Mapping[str, Any], ...] = ()

    @property
    def created_at(self) -> datetime:
        return _parse_utc(self.created_utc)

    @property
    def clean(self) -> bool:
        return bool(self.saved_document_hash) and (
            self.document_hash == self.saved_document_hash
        )

    @property
    def source_state(self) -> RecoveryState:
        if self.source_path is None:
            return "unsaved"
        if not self.source_path.exists():
            return "source_missing"
        try:
            stat = self.source_path.stat()
            if (
                self.source_sha256
                and stat.st_size == self.source_size
                and _sha256_file(self.source_path) == self.source_sha256
            ):
                return "source_unchanged"
            if self.source_sha256:
                return "source_changed"
            unchanged = (
                stat.st_size == self.source_size
                and stat.st_mtime_ns == self.source_mtime_ns
            )
            return "source_unchanged" if unchanged else "source_changed"
        except OSError:
            return "source_unavailable"

    @property
    def recommendation(self) -> str:
        if self.clean:
            return "discard_clean"
        state = self.source_state
        if state in ("source_changed", "source_unavailable"):
            return "open_copy"
        if state in ("unsaved", "source_missing"):
            return "restore_as"
        return "restore"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_SCHEMA,
            "version": RECOVERY_VERSION,
            "document_id": self.document_id,
            "revision": {
                "id": self.revision_id,
                "sequence": self.revision_sequence,
                "document_hash": self.document_hash,
                "model_hash": self.model_hash,
            },
            "saved_document_hash": self.saved_document_hash,
            "created_utc": self.created_utc,
            "snapshot_uri": _SNAPSHOT_NAME,
            "snapshot_sha256": self.snapshot_sha256,
            "source": None
            if self.source_path is None
            else {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
                "size": self.source_size,
                "mtime_ns": self.source_mtime_ns,
            },
            "artifacts": [dict(value) for value in self.artifact_refs],
        }


def default_recovery_root() -> Path:
    return Path(user_state_path("ANYfem", appauthor=False)) / "recovery"


def write_autosave(
    document: Mapping[str, Any],
    *,
    document_id: str,
    revision_id: str,
    revision_sequence: int,
    document_hash: str,
    model_hash: str,
    saved_document_hash: str = "",
    source_path: str | Path | None = None,
    artifact_refs: Iterable[Mapping[str, Any]] = (),
    root: str | Path | None = None,
    created_at: datetime | None = None,
    keep: int = 3,
    max_age: timedelta = timedelta(days=7),
) -> RecoveryCandidate:
    """Commit one self-contained autosave without touching the source file."""

    if not str(document_id):
        raise RecoveryError("autosave needs a document ID")
    if not str(revision_id):
        raise RecoveryError("autosave needs a revision ID")
    if int(revision_sequence) < 0:
        raise RecoveryError("revision sequence cannot be negative")
    now = (created_at or _utc_now()).astimezone(timezone.utc)
    recovery_root = _resolved(root or default_recovery_root())
    source = None if source_path is None else _resolved(source_path)
    doc_key = _short_key(str(document_id))
    source_key = "unsaved" if source is None else _short_key(str(source).casefold())
    container = _safe_child(recovery_root, doc_key, source_key)
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    bundle_name = (
        f"{stamp}-{int(revision_sequence):012d}-"
        f"{_short_key(str(revision_id), 12)}-{uuid4().hex[:8]}"
    )
    final = _safe_child(container, bundle_name)
    staging = _safe_child(container, f".{bundle_name}.{uuid4().hex}.tmp")
    source_info = _source_identity(source)
    snapshot_payload = _json_bytes(document, pretty=True)
    candidate = RecoveryCandidate(
        bundle_path=final,
        snapshot_path=final / _SNAPSHOT_NAME,
        document_id=str(document_id),
        revision_id=str(revision_id),
        revision_sequence=int(revision_sequence),
        document_hash=str(document_hash),
        model_hash=str(model_hash),
        saved_document_hash=str(saved_document_hash),
        created_utc=_utc_text(now),
        source_path=source,
        source_sha256=source_info.get("sha256"),
        source_size=source_info.get("size"),
        source_mtime_ns=source_info.get("mtime_ns"),
        snapshot_sha256=_sha256_bytes(snapshot_payload),
        artifact_refs=tuple(dict(value) for value in artifact_refs),
    )
    container.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        _atomic_write(staging / _SNAPSHOT_NAME, snapshot_payload)
        _atomic_write(
            staging / _METADATA_NAME,
            _json_bytes(candidate.to_dict(), pretty=True),
        )
        os.replace(staging, final)
    except BaseException:
        if staging.exists():
            _safe_rmtree(staging, recovery_root)
        raise
    committed = _read_candidate(final, recovery_root)
    prune_recoveries(
        root=recovery_root,
        document_id=str(document_id),
        source_path=source,
        keep=keep,
        max_age=max_age,
        now=now,
    )
    return committed


def discover_recoveries(
    *,
    root: str | Path | None = None,
    max_age: timedelta = timedelta(days=7),
    include_expired: bool = False,
    latest_only: bool = True,
    now: datetime | None = None,
) -> list[RecoveryCandidate]:
    recovery_root = _resolved(root or default_recovery_root())
    if not recovery_root.is_dir():
        return []
    current = (now or _utc_now()).astimezone(timezone.utc)
    candidates: list[RecoveryCandidate] = []
    for bundle in _bundle_directories(recovery_root):
        try:
            candidate = _read_candidate(bundle, recovery_root)
        except RecoveryError:
            continue
        if not include_expired and current - candidate.created_at > max_age:
            continue
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (item.created_at, item.revision_sequence), reverse=True
    )
    if not latest_only:
        return candidates
    selected: list[RecoveryCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (
            candidate.document_id,
            "" if candidate.source_path is None else str(candidate.source_path).casefold(),
        )
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
    return selected


def load_recovery(candidate: RecoveryCandidate) -> dict[str, Any]:
    try:
        if _sha256_file(candidate.snapshot_path) != candidate.snapshot_sha256:
            raise RecoveryError("recovery snapshot checksum does not match metadata")
    except OSError as error:
        raise RecoveryError(f"cannot read recovery snapshot: {error}") from None
    try:
        payload = json.loads(candidate.snapshot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read recovery snapshot: {error}") from None
    if not isinstance(payload, dict):
        raise RecoveryError("recovery snapshot must be a JSON object")
    return payload


def discard_recovery(
    candidate: RecoveryCandidate, *, root: str | Path | None = None
) -> None:
    recovery_root = _resolved(root or default_recovery_root())
    _safe_rmtree(candidate.bundle_path, recovery_root)
    _remove_empty_parents(candidate.bundle_path.parent, recovery_root)


def prune_recoveries(
    *,
    root: str | Path | None = None,
    document_id: str | None = None,
    source_path: str | Path | None | object = _SOURCE_UNSET,
    keep: int = 3,
    max_age: timedelta = timedelta(days=7),
    now: datetime | None = None,
) -> list[Path]:
    """Remove expired/excess snapshots, scoped to a document when requested."""

    if keep < 1:
        raise ValueError("keep must be at least one")
    recovery_root = _resolved(root or default_recovery_root())
    current = (now or _utc_now()).astimezone(timezone.utc)
    source_filter = source_path is not _SOURCE_UNSET
    source = (
        None
        if source_path is _SOURCE_UNSET or source_path is None
        else _resolved(source_path)  # type: ignore[arg-type]
    )
    candidates = discover_recoveries(
        root=recovery_root,
        include_expired=True,
        latest_only=False,
        now=current,
    )
    groups: dict[tuple[str, str], list[RecoveryCandidate]] = {}
    for candidate in candidates:
        if document_id is not None and candidate.document_id != str(document_id):
            continue
        if source_filter and candidate.source_path != source:
            continue
        key = (
            candidate.document_id,
            "" if candidate.source_path is None else str(candidate.source_path).casefold(),
        )
        groups.setdefault(key, []).append(candidate)
    removed: list[Path] = []
    for items in groups.values():
        items.sort(
            key=lambda item: (item.created_at, item.revision_sequence), reverse=True
        )
        for index, candidate in enumerate(items):
            expired = current - candidate.created_at > max_age
            if expired or index >= keep:
                _safe_rmtree(candidate.bundle_path, recovery_root)
                removed.append(candidate.bundle_path)
    for path in removed:
        _remove_empty_parents(path.parent, recovery_root)
    return removed


def _source_identity(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        stat = path.stat()
        return {
            "sha256": _sha256_file(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {}


def _read_candidate(bundle: Path, root: Path) -> RecoveryCandidate:
    resolved_bundle = bundle.resolve(strict=False)
    try:
        resolved_bundle.relative_to(root.resolve(strict=False))
    except ValueError:
        raise RecoveryError("recovery bundle escapes its root") from None
    metadata_path = _safe_child(resolved_bundle, _METADATA_NAME)
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"invalid recovery metadata: {error}") from None
    try:
        if data.get("schema") != RECOVERY_SCHEMA:
            raise ValueError("not an ANYfem recovery record")
        if int(data.get("version", 0)) != RECOVERY_VERSION:
            raise ValueError("unsupported recovery version")
        if data.get("snapshot_uri") != _SNAPSHOT_NAME:
            raise ValueError("unsafe recovery snapshot URI")
        snapshot = _safe_child(resolved_bundle, _SNAPSHOT_NAME)
        if not snapshot.is_file():
            raise ValueError("recovery snapshot is missing")
        snapshot_sha256 = str(data["snapshot_sha256"])
        if not snapshot_sha256.startswith("sha256:"):
            raise ValueError("recovery snapshot checksum is invalid")
        if _sha256_file(snapshot) != snapshot_sha256:
            raise ValueError("recovery snapshot checksum does not match metadata")
        revision = data["revision"]
        source = data.get("source")
        source_path = None if source is None else _resolved(source["path"])
        created = str(data["created_utc"])
        _parse_utc(created)
        document_id = str(data["document_id"])
        revision_id = str(revision["id"])
        if not document_id or not revision_id:
            raise ValueError("recovery identity is incomplete")
        artifacts = data.get("artifacts", ())
        if not isinstance(artifacts, list) or any(
            not isinstance(item, Mapping) for item in artifacts
        ):
            raise ValueError("recovery artifacts must be objects")
        return RecoveryCandidate(
            bundle_path=resolved_bundle,
            snapshot_path=snapshot,
            document_id=document_id,
            revision_id=revision_id,
            revision_sequence=int(revision["sequence"]),
            document_hash=str(revision["document_hash"]),
            model_hash=str(revision["model_hash"]),
            saved_document_hash=str(data.get("saved_document_hash", "")),
            created_utc=created,
            source_path=source_path,
            source_sha256=None if source is None else source.get("sha256"),
            source_size=None if source is None else _optional_int(source.get("size")),
            source_mtime_ns=(
                None if source is None else _optional_int(source.get("mtime_ns"))
            ),
            snapshot_sha256=snapshot_sha256,
            artifact_refs=tuple(dict(item) for item in artifacts),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RecoveryError(f"invalid recovery metadata: {error}") from None


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _bundle_directories(root: Path) -> Iterable[Path]:
    # The fixed three-level layout avoids following arbitrary paths from
    # metadata and avoids a broad recursive traversal of the state directory.
    for document_dir in root.iterdir():
        if not document_dir.is_dir() or document_dir.is_symlink():
            continue
        for source_dir in document_dir.iterdir():
            if not source_dir.is_dir() or source_dir.is_symlink():
                continue
            for bundle in source_dir.iterdir():
                if (
                    bundle.is_dir()
                    and not bundle.is_symlink()
                    and not bundle.name.startswith(".")
                    and (bundle / _METADATA_NAME).is_file()
                ):
                    yield bundle


def _safe_rmtree(path: Path, root: Path) -> None:
    resolved = path.resolve(strict=False)
    base = root.resolve(strict=False)
    try:
        relative = resolved.relative_to(base)
    except ValueError:
        raise RecoveryError("refusing to remove outside recovery root") from None
    if not relative.parts or resolved == base:
        raise RecoveryError("refusing to remove the recovery root")
    if resolved.exists():
        shutil.rmtree(resolved)


def _remove_empty_parents(path: Path, root: Path) -> None:
    base = root.resolve(strict=False)
    current = path.resolve(strict=False)
    while current != base:
        try:
            current.relative_to(base)
        except ValueError:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent

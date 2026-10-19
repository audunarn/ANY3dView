"""Immutable HDF5 mesh/result sidecars and lazy result access."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np

from ..model.records import ArtifactRef, ResultQuantityDescriptor

__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "LazyResultDataset",
    "ResultField",
]


class ArtifactError(ValueError):
    """Raised when an artifact is missing, unsafe, corrupt or incompatible."""


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ArtifactError(
            "HDF5 artifact support requires h5py>=3.15,<4"
        ) from None
    return h5py


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.tolist() if hasattr(item, "tolist") else item,
    )


class ArtifactStore:
    """Sidecar directory belonging to one ``.anyfem`` project path."""

    RESULT_SCHEMA = "anyfem.result"
    MESH_SCHEMA = "anyfem.mesh"
    VERSION = 1

    def __init__(self, project_path: str | Path) -> None:
        project = Path(project_path)
        if not project.suffix:
            project = project.with_suffix(".anyfem")
        self.project_path = project
        self.root = project.with_name(project.name + "-data")

    def ensure(self) -> Path:
        for name in ("meshes", "results", "logs"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self.root

    def resolve(self, uri: str) -> Path:
        relative = Path(uri.replace("/", os.sep))
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError("artifact URI must stay inside the project data root")
        root = self.root.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ArtifactError("artifact URI escapes the project data root") from None
        return target

    def write_mesh(
        self,
        mesh,
        *,
        mesh_id: str | None = None,
        document_id: str = "",
        model_hash: str = "",
        mesh_hash: str = "",
        structural_preparation: Mapping[str, Any] | None = None,
        imported_model: Mapping[str, Any] | None = None,
        embedded_source: bytes | bytearray | memoryview | None = None,
    ) -> ArtifactRef:
        from anymesher.serialize import mesh_to_dict

        imported_payload, source_contents = _prepare_imported_model(
            imported_model, embedded_source
        )
        identifier = str(mesh_id or uuid4())
        uri = f"meshes/{identifier}.anymesh.h5"
        destination = self.resolve(uri)
        self.ensure()
        h5py = _h5py()
        with _atomic_destination(destination) as temporary:
            with h5py.File(temporary, "w") as handle:
                _root_attributes(
                    handle,
                    schema=self.MESH_SCHEMA,
                    artifact_id=identifier,
                    document_id=document_id,
                )
                handle.attrs["model_hash"] = model_hash
                handle.attrs["mesh_hash"] = mesh_hash
                group = handle.create_group("mesh")
                node_ids = np.asarray(sorted(mesh.nodes), dtype=np.int64)
                xyz = np.asarray([mesh.nodes[int(node)] for node in node_ids], dtype=np.float64)
                nodes = group.create_group("nodes")
                _dataset(nodes, "ids", node_ids)
                _dataset(nodes, "xyz", xyz)
                elements = group.create_group("elements")
                for name in ("quads", "tris", "beams"):
                    mapping = getattr(mesh, name)
                    family = elements.create_group(name)
                    ids = np.asarray(sorted(mapping), dtype=np.int64)
                    connectivity = np.asarray([mapping[int(key)] for key in ids], dtype=np.int64)
                    _dataset(family, "ids", ids)
                    _dataset(family, "connectivity", connectivity)
                # The owner codec remains authoritative for associations and
                # provides forward-compatible reconstruction alongside the
                # columnar arrays used for fast inspection.
                codec = np.frombuffer(_json(mesh_to_dict(mesh)).encode("utf-8"), dtype=np.uint8)
                _dataset(group, "codec_json", codec)
                group.create_dataset(
                    "structural_preparation_json",
                    data=_json(dict(structural_preparation or {})),
                    dtype=h5py.string_dtype("utf-8"),
                )
                if imported_payload is not None:
                    imported = handle.create_group("imported_model")
                    imported.create_dataset(
                        "json",
                        data=_json(imported_payload),
                        dtype=h5py.string_dtype("utf-8"),
                    )
                    if source_contents is not None:
                        source = _dataset(
                            imported,
                            "source_bytes",
                            np.frombuffer(source_contents, dtype=np.uint8),
                        )
                        source_metadata = imported_payload["source"]
                        source.attrs["name"] = source_metadata["name"]
                        source.attrs["format"] = source_metadata["format"]
                        source.attrs["sha256"] = source_metadata["sha256"]
                handle.attrs["complete"] = True
                handle.flush()
        return _artifact(uri, identifier, "mesh", destination)

    def read_mesh(self, artifact: ArtifactRef | str):
        from anymesher.serialize import mesh_from_dict

        path = self._path_for_read(artifact, "mesh")
        h5py = _h5py()
        try:
            with h5py.File(path, "r") as handle:
                self._validate_root(handle, self.MESH_SCHEMA)
                self._validate_artifact_identity(handle, artifact)
                raw = bytes(
                    np.asarray(handle["mesh/codec_json"], dtype=np.uint8)
                ).decode("utf-8")
            payload = json.loads(raw)
            return mesh_from_dict(payload)
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            UnicodeError,
            json.JSONDecodeError,
        ) as error:
            raise ArtifactError(f"cannot read mesh artifact {path}: {error}") from None

    def read_mesh_metadata(
        self, artifact: ArtifactRef | str
    ) -> dict[str, Any]:
        """Read verified mesh provenance without materializing the mesh.

        Imported-source bytes stay lazy; :meth:`read_embedded_source` returns
        them.  Their name, format, size and checksum are still validated here
        so callers never make a recovery decision from untrusted metadata.
        """

        path = self._path_for_read(artifact, "mesh")
        h5py = _h5py()
        try:
            with h5py.File(path, "r") as handle:
                self._validate_root(handle, self.MESH_SCHEMA)
                self._validate_artifact_identity(handle, artifact)
                metadata: dict[str, Any] = {
                    "schema": _text_attribute(handle.attrs.get("schema", "")),
                    "version": int(handle.attrs.get("version", 0)),
                    "artifact_id": _text_attribute(
                        handle.attrs.get("artifact_id", "")
                    ),
                    "document_id": _text_attribute(
                        handle.attrs.get("document_id", "")
                    ),
                    "created_utc": _text_attribute(
                        handle.attrs.get("created_utc", "")
                    ),
                    "model_hash": _text_attribute(
                        handle.attrs.get("model_hash", "")
                    ),
                    "mesh_hash": _text_attribute(
                        handle.attrs.get("mesh_hash", "")
                    ),
                    "byte_size": path.stat().st_size,
                    "imported_model": None,
                    "embedded_source": None,
                    "structural_preparation": {},
                }
                if "mesh/structural_preparation_json" in handle:
                    raw_preparation = handle[
                        "mesh/structural_preparation_json"
                    ][()]
                    preparation = json.loads(
                        _text_attribute(raw_preparation)
                    )
                    if not isinstance(preparation, dict):
                        raise ArtifactError(
                            "mesh structural preparation must be a JSON object"
                        )
                    metadata["structural_preparation"] = preparation
                if "imported_model" in handle:
                    imported = handle["imported_model"]
                    payload = _read_json_dataset(imported, "json")
                    if not isinstance(payload, Mapping):
                        raise ArtifactError(
                            "imported-model metadata must be a JSON object"
                        )
                    payload = dict(payload)
                    metadata["imported_model"] = payload
                    if "source_bytes" in imported:
                        _contents, source = _read_embedded_source(
                            imported, payload
                        )
                        metadata["embedded_source"] = source
                return metadata
        except ArtifactError:
            raise
        except (OSError, KeyError, TypeError, ValueError, UnicodeError) as error:
            raise ArtifactError(
                f"cannot read mesh metadata {path}: {error}"
            ) from None

    def read_embedded_source(self, artifact: ArtifactRef | str) -> bytes:
        """Return checksum-verified source bytes embedded in a mesh artifact."""

        path = self._path_for_read(artifact, "mesh")
        h5py = _h5py()
        try:
            with h5py.File(path, "r") as handle:
                self._validate_root(handle, self.MESH_SCHEMA)
                self._validate_artifact_identity(handle, artifact)
                imported = handle["imported_model"]
                payload = _read_json_dataset(imported, "json")
                if not isinstance(payload, Mapping):
                    raise ArtifactError(
                        "imported-model metadata must be a JSON object"
                    )
                contents, _metadata = _read_embedded_source(
                    imported, payload
                )
                return contents
        except ArtifactError:
            raise
        except (OSError, KeyError, TypeError, ValueError, UnicodeError) as error:
            raise ArtifactError(
                f"cannot read embedded source from {path}: {error}"
            ) from None

    def write_result(
        self,
        *,
        job_id: str,
        document_id: str,
        mesh_id: str,
        model_hash: str,
        mesh_hash: str,
        analysis_hash: str,
        fields: Mapping[str, tuple[ResultQuantityDescriptor, Any]],
        frames: Sequence[float] = (),
        frame_kind: str = "static",
        histories: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
        tables: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        summary: Mapping[str, Any] | None = None,
        diagnostics: Sequence[Any] = (),
        partial: bool = False,
    ) -> ArtifactRef:
        identifier = str(job_id)
        uri = f"results/{identifier}.anyres.h5"
        destination = self.resolve(uri)
        self.ensure()
        h5py = _h5py()
        with _atomic_destination(destination) as temporary:
            with h5py.File(temporary, "w") as handle:
                _root_attributes(
                    handle,
                    schema=self.RESULT_SCHEMA,
                    artifact_id=identifier,
                    document_id=document_id,
                )
                handle.attrs["model_hash"] = model_hash
                handle.attrs["mesh_hash"] = mesh_hash
                handle.attrs["analysis_hash"] = analysis_hash
                handle.attrs["mesh_id"] = mesh_id
                handle.attrs["partial"] = bool(partial)
                frame_group = handle.create_group("frames")
                frame_group.attrs["kind"] = frame_kind
                _dataset(frame_group, "values", np.asarray(frames, dtype=np.float64))
                field_group = handle.create_group("fields")
                for key, (descriptor, values) in fields.items():
                    group = field_group.create_group(str(key))
                    array = np.asarray(values)
                    _dataset(group, "values", array, frame_major=array.ndim > 1)
                    for attr, value in descriptor.to_dict().items():
                        group.attrs[attr] = _json(value) if isinstance(value, (dict, list)) else value
                history_group = handle.create_group("histories")
                for key, (x_values, y_values) in (histories or {}).items():
                    group = history_group.create_group(str(key))
                    _dataset(group, "x", np.asarray(x_values, dtype=np.float64))
                    _dataset(group, "y", np.asarray(y_values, dtype=np.float64))
                table_group = handle.create_group("tables")
                for key, value in (tables or {}).items():
                    array = np.asarray(value)
                    if array.dtype.kind in "OUS":
                        table_group.create_dataset(str(key), data=_json(value), dtype=h5py.string_dtype("utf-8"))
                    else:
                        _dataset(table_group, str(key), array)
                metadata = handle.create_group("metadata")
                string_type = h5py.string_dtype("utf-8")
                metadata.create_dataset("provenance_json", data=_json(provenance or {}), dtype=string_type)
                metadata.create_dataset("summary_json", data=_json(summary or {}), dtype=string_type)
                metadata.create_dataset("diagnostics_json", data=_json(list(diagnostics)), dtype=string_type)
                handle.attrs["complete"] = True
                handle.flush()
        return _artifact(uri, identifier, "result", destination)

    def write_log(
        self,
        job_id: str,
        entries: Sequence[Mapping[str, Any]],
    ) -> ArtifactRef:
        """Atomically persist a structured, human-readable JSONL job log."""

        job_identifier = str(job_id)
        # Result artifacts historically use the job UUID as their artifact
        # UUID.  A log is a second artifact and therefore needs its own stable
        # identity or it would overwrite the result entry in Project.artifacts.
        identifier = str(uuid5(NAMESPACE_URL, f"anyfem:job-log:{job_identifier}"))
        uri = f"logs/{job_identifier}.log"
        destination = self.resolve(uri)
        self.ensure()
        encoded = "".join(_json(dict(entry)) + "\n" for entry in entries).encode(
            "utf-8"
        )
        with _atomic_plain_destination(destination) as temporary:
            with temporary.open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            # Reparse the exact bytes before they are allowed to replace an
            # earlier valid log reference.
            _decode_log(temporary.read_text(encoding="utf-8"))
        return _artifact(uri, identifier, "log", destination)

    def read_log(
        self, artifact: ArtifactRef | str
    ) -> tuple[dict[str, Any], ...]:
        path = self._path_for_read(artifact, "log")
        try:
            return _decode_log(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise ArtifactError(f"cannot read job log {path}: {error}") from None

    def open_result(self, artifact: ArtifactRef | str, *, defer=False) -> "LazyResultDataset":
        if defer:
            from copy import deepcopy
            snapshot=deepcopy(artifact)
            uri=snapshot.uri if isinstance(snapshot,ArtifactRef) else snapshot
            return LazyResultDataset(self.resolve(uri),validator=lambda:self._path_for_read(snapshot,"result"))
        dataset = LazyResultDataset(self._path_for_read(artifact, "result"))
        dataset.validate()
        return dataset

    def verify(self, artifact: ArtifactRef) -> bool:
        if not artifact.sha256:
            return False
        try:
            self._path_for_read(artifact, artifact.kind)
        except (ArtifactError, OSError):
            return False
        return True

    def copy_from(
        self, source: "ArtifactStore", artifact: ArtifactRef
    ) -> ArtifactRef:
        """Atomically copy one verified sidecar for Save As/relocation."""

        if not isinstance(source, ArtifactStore):
            raise TypeError("source must be an ArtifactStore")
        source_path = source._path_for_read(artifact, artifact.kind)
        destination = self.resolve(artifact.uri)
        if source_path.resolve() == destination.resolve():
            return artifact
        self.ensure()
        atomic_writer = (
            _atomic_plain_destination if artifact.kind == "log" else _atomic_destination
        )
        with atomic_writer(destination) as temporary:
            shutil.copyfile(source_path, temporary)
            if artifact.sha256 and _sha256(temporary) != artifact.sha256:
                raise ArtifactError(
                    f"copied artifact checksum mismatch for {destination.name}"
                )
            if artifact.kind == "log":
                _decode_log(temporary.read_text(encoding="utf-8"))
        copied = _artifact(artifact.uri, artifact.id, artifact.kind, destination)
        return copied

    def _path_for_read(
        self, artifact: ArtifactRef | str, expected_kind: str
    ) -> Path:
        uri = artifact.uri if isinstance(artifact, ArtifactRef) else artifact
        path = self.resolve(uri)
        if not path.is_file():
            raise ArtifactError(f"artifact file is missing: {path}")
        if not isinstance(artifact, ArtifactRef):
            return path
        if artifact.kind not in ("", "unknown", expected_kind):
            raise ArtifactError(
                f"expected a {expected_kind} artifact, got {artifact.kind!r}"
            )
        if artifact.byte_size and path.stat().st_size != artifact.byte_size:
            raise ArtifactError(
                f"artifact size mismatch for {path.name}; the sidecar is corrupt "
                "or belongs to another project revision"
            )
        if artifact.sha256:
            expected = _validated_sha256(
                artifact.sha256, f"artifact {artifact.id} checksum"
            )
            actual = _sha256(path)
            if actual != expected:
                raise ArtifactError(
                    f"artifact checksum mismatch for {path.name}; the sidecar "
                    "is corrupt or belongs to another project revision"
                )
        return path

    @staticmethod
    def _validate_artifact_identity(handle, artifact: ArtifactRef | str) -> None:
        if not isinstance(artifact, ArtifactRef):
            return
        actual = _text_attribute(handle.attrs.get("artifact_id", ""))
        if actual != artifact.id:
            raise ArtifactError(
                f"artifact ID mismatch: project expects {artifact.id!r}, "
                f"sidecar contains {actual!r}"
            )

    def _validate_root(self, handle, expected_schema: str) -> None:
        if handle.attrs.get("schema") != expected_schema:
            raise ArtifactError(f"expected {expected_schema} artifact")
        if int(handle.attrs.get("version", 0)) > self.VERSION:
            raise ArtifactError("artifact was written by a newer ANYfem")
        if not bool(handle.attrs.get("complete", False)):
            raise ArtifactError("artifact is incomplete")


class ResultField:
    def __init__(self, dataset: "LazyResultDataset", key: str) -> None:
        self.dataset = dataset
        self.key = key

    @property
    def descriptor(self) -> ResultQuantityDescriptor:
        if self.dataset._header is not None:
            attributes=dict(self.dataset._header["originals"][self.key])
        else:
            h5py = _h5py()
            with h5py.File(self.dataset.path, "r") as handle:
                group = handle[f"fields/{self.key}"]
                attributes = dict(group.attrs)
        def decoded(name: str, default):
            value = attributes.get(name, default)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str) and value[:1] in ("[", "{"):
                return json.loads(value)
            return value
        return ResultQuantityDescriptor(
            key=str(decoded("key", self.key)),
            label=str(decoded("label", self.key)),
            location=str(decoded("location", "global")),
            unit=str(decoded("unit", "")),
            components=tuple(decoded("components", ())),
            basis=str(decoded("basis", "global")),
            frames=tuple(float(value) for value in decoded("frames", ())),
            recovery=str(decoded("recovery", "native")),
            reduction=str(decoded("reduction", "none")),
            deformation_required=bool(decoded("deformation_required", False)),
            provenance=dict(decoded("provenance", {})),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        if self.dataset._header is not None:return tuple(self.dataset._header["descriptors"][self.key]["stored_shape"])
        h5py = _h5py()
        with h5py.File(self.dataset.path, "r") as handle:
            return tuple(handle[f"fields/{self.key}/values"].shape)

    def read_component(self, component, frame=0):
        reader=self.dataset.result_reader()
        descriptor=next(d for d in reader.components(self.key) if d["component"]==component)
        cases=descriptor.get("case_ids",tuple(c["case_id"] for c in reader.cases()))
        index=0 if frame is None else int(frame)
        if index<0 or index>=len(cases):raise IndexError("quantity-local frame is unavailable")
        values=reader.read_stored_field(cases[index],self.key,component)
        return np.asarray(values).reshape(tuple(descriptor["component_shape"]))

    def root_frame_index(self,index):
        """Resolve a GUI root frame to a proved quantity-local row."""
        from anypost.errors import UnavailableFieldError
        root=tuple(self.dataset.frames);index=int(index)
        if index<0 or (root and index>=len(root)):raise UnavailableFieldError("requested root frame is unavailable")
        descriptor=self.descriptor
        frames=tuple(descriptor.frames)
        if descriptor.provenance.get("frame_policy")=="envelope":
            return 0  # explicitly synthetic quantity; never a different source frame
        if frames:
            if not root:
                if index>=len(frames):raise UnavailableFieldError("requested quantity frame is unavailable")
                return index
            # A whole matching sequence proves row identity even when modal
            # frequencies or load factors repeat. Do not infer ambiguous subsets.
            if len(frames)==len(root) and np.array_equal(np.asarray(frames),np.asarray(root),equal_nan=True):return index
            value=root[index]
            matches=[i for i,current in enumerate(frames) if current==value]
            if len(matches)!=1:raise UnavailableFieldError(f"{self.key}: no quantity row at root frame {index}")
            return matches[0]
        transport=self.dataset.result_reader().components(self.key)[0]
        if len(self.shape)==1 and tuple(transport["component_shape"])==self.shape:
            if len(root)>1:raise UnavailableFieldError(f"{self.key}: source frame association is undeclared")
            return None
        cases=tuple(transport.get("case_ids",tuple(c["case_id"] for c in self.dataset.result_reader().cases())))
        if str(index) not in cases:raise UnavailableFieldError(f"{self.key}: no proven row at root frame {index}")
        return cases.index(str(index))

    def read(self, frame: int | slice | None = None) -> np.ndarray:
        from ..post.access import read_sidecar_field
        return read_sidecar_field(self.dataset.path, self.key, self.shape, frame, reader=self.dataset.result_reader())



class LazyResultDataset:
    def __init__(self, path: str | Path, *, validator=None) -> None:
        self._path = Path(path)
        self._validator=validator
        self._header=None
        from threading import RLock
        self._reader_lock = RLock()
        self._post_reader = None
        self._closed = False

    @property
    def path(self):
        if self._validator is not None:
            validated=self._validator()
            self._path=Path(validated)
            self._validator=None
        return self._path

    def result_reader(self):
        from anypost import open_results
        from ..post.access import CACHE_BYTES, WORKING_BYTES
        with self._reader_lock:
            if self._closed:
                from anypost.errors import ClosedReaderError
                raise ClosedReaderError("result dataset is retired")
            if self._post_reader is None:
                if self._validator is not None:
                    self.path  # validates and retains the admitted path
                # One active desktop reader leaves room for old/new published
                # Field/scalar buffers, each admitted by the query-work cap.
                self._post_reader = open_results(self.path,cache_bytes=CACHE_BYTES-WORKING_BYTES,working_bytes=WORKING_BYTES)
            return self._post_reader

    @property
    def metadata_ready(self):return self._header is not None

    def prepare_metadata(self):
        """Open/verify and snapshot source headers on the caller's I/O worker."""
        reader=self.result_reader()
        metadata=reader.metadata()
        identity=self.identity
        self._header=dict(identity=identity,metadata=metadata,fields=tuple(sorted({d["quantity_id"] for d in reader.components()})),
                          descriptors={d["quantity_id"]:d for d in reader.components()},
                          originals={d["source_key"]:d for d in metadata.get("original_descriptors",())},
                          frames=reader.frames(),tables=reader.table_keys())
        return self

    def close(self):
        with self._reader_lock:
            self._closed=True
            self.release_reader()

    def release_reader(self):
        """Evict an inactive transport cache without retiring the dataset."""
        with self._reader_lock:
            if self._post_reader is not None:
                self._post_reader.close()
                self._post_reader=None

    def validate(self) -> None:
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as handle:
                if handle.attrs.get("schema") != ArtifactStore.RESULT_SCHEMA:
                    raise ArtifactError("not an ANYfem result artifact")
                if int(handle.attrs.get("version", 0)) > ArtifactStore.VERSION:
                    raise ArtifactError("result artifact is newer than this ANYfem")
                if not bool(handle.attrs.get("complete", False)):
                    raise ArtifactError("result artifact is incomplete")
        except OSError as error:
            raise ArtifactError(f"cannot open result artifact {self.path}: {error}") from None

    @property
    def field_keys(self) -> tuple[str, ...]:
        if self._header is not None:return self._header["fields"]
        h5py = _h5py()
        with h5py.File(self.path, "r") as handle:
            return tuple(sorted(handle["fields"].keys()))

    def field(self, key: str) -> ResultField:
        if key not in self.field_keys:
            raise KeyError(f"result has no field {key!r}")
        return ResultField(self, key)

    @property
    def frames(self) -> np.ndarray:
        if self._header is not None:return self._header["frames"]
        h5py = _h5py()
        with h5py.File(self.path, "r") as handle:
            return np.asarray(handle["frames/values"], dtype=float)

    @property
    def identity(self) -> dict[str, Any]:
        """Return immutable submission identity without reading result arrays.

        The values originate in root HDF5 attributes written before the
        artifact is committed.  Keeping this separate from ``provenance``
        makes report and automation callers independent of solver-specific
        metadata while still allowing them to reproduce the exact submission.
        """

        if self._header is not None:return dict(self._header["identity"])
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as handle:
                self._validate_open_handle(handle)
                frame_group = handle.get("frames")
                return {
                    "schema": _text_attribute(handle.attrs.get("schema", "")),
                    "schema_version": int(handle.attrs.get("version", 0)),
                    "artifact_id": _text_attribute(
                        handle.attrs.get("artifact_id", "")
                    ),
                    "document_id": _text_attribute(
                        handle.attrs.get("document_id", "")
                    ),
                    "mesh_id": _text_attribute(handle.attrs.get("mesh_id", "")),
                    "model_hash": _text_attribute(
                        handle.attrs.get("model_hash", "")
                    ),
                    "mesh_hash": _text_attribute(
                        handle.attrs.get("mesh_hash", "")
                    ),
                    "analysis_hash": _text_attribute(
                        handle.attrs.get("analysis_hash", "")
                    ),
                    "created_utc": _text_attribute(
                        handle.attrs.get("created_utc", "")
                    ),
                    "partial": bool(handle.attrs.get("partial", False)),
                    "frame_kind": (
                        ""
                        if frame_group is None
                        else _text_attribute(frame_group.attrs.get("kind", ""))
                    ),
                }
        except ArtifactError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(
                f"cannot read result identity from {self.path}: {error}"
            ) from None

    def history(
        self, key: str, rows: int | slice | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        from anypost import open_results
        from ..post.access import CACHE_BYTES, WORKING_BYTES
        return self.result_reader().read_history(key, rows=rows)

    def history_shape(self, key: str) -> tuple[int, ...]:
        if self._header is not None:
            try:return tuple(self._header["metadata"]["history_shapes"][key])
            except (KeyError,TypeError,ValueError) as error:raise ArtifactError(f"cannot inspect result history {key!r}: {error}") from None
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as handle:
                group = handle[f"histories/{key}"]
                x_shape = tuple(group["x"].shape)
                y_shape = tuple(group["y"].shape)
                if x_shape != y_shape:
                    raise ArtifactError(
                        f"history {key!r} has mismatched x/y shapes "
                        f"{x_shape!r} and {y_shape!r}"
                    )
                return x_shape
        except ArtifactError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(
                f"cannot inspect result history {key!r}: {error}"
            ) from None

    @property
    def history_keys(self) -> tuple[str, ...]:
        if self._header is not None:return tuple(self._header["metadata"].get("history_keys",()))
        h5py = _h5py()
        with h5py.File(self.path, "r") as handle:
            return tuple(sorted(handle["histories"].keys()))

    @property
    def table_keys(self) -> tuple[str, ...]:
        if self._header is not None:return self._header["tables"]
        h5py = _h5py()
        with h5py.File(self.path, "r") as handle:
            return tuple(sorted(handle["tables"].keys()))

    def _source_table(self, key: str, rows: int | slice | None = None) -> Any:
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as handle:
                values = handle[f"tables/{key}"]
                # JSON tables are scalar strings and cannot be sliced.  Dense
                # numeric association/result tables can be previewed without
                # materializing every row.
                if rows is None or values.ndim == 0:
                    raw = values[()]
                else:
                    raw = values[rows]
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"cannot read result table {key!r}: {error}") from None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as error:
                raise ArtifactError(
                    f"result table {key!r} contains invalid JSON: {error}"
                ) from None
        return np.asarray(raw)

    def table(self, key: str, rows: int | slice | None = None) -> Any:
        from anypost import open_results
        from ..post.access import CACHE_BYTES, WORKING_BYTES
        return self.result_reader().table(key, rows)

    def table_shape(self, key: str) -> tuple[int, ...]:
        if self._header is not None:
            try:return tuple(self._header["metadata"]["table_shapes"][key])
            except (KeyError,TypeError,ValueError) as error:raise ArtifactError(f"cannot inspect result table {key!r}: {error}") from None
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as handle:
                return tuple(handle[f"tables/{key}"].shape)
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(
                f"cannot inspect result table {key!r}: {error}"
            ) from None

    def metadata(self, name: str) -> Any:
        if self._header is not None:
            try:return self._header["metadata"]["provenance"][name]
            except (KeyError,TypeError,ValueError) as error:raise ArtifactError(f"cannot read result metadata {name!r}: {error}") from None
        h5py = _h5py()
        try:
            with h5py.File(self.path, "r") as handle:
                raw = handle[f"metadata/{name}_json"][()]
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ArtifactError(
                f"cannot read result metadata {name!r}: {error}"
            ) from None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError as error:
            raise ArtifactError(
                f"result metadata {name!r} contains invalid JSON: {error}"
            ) from None

    @staticmethod
    def _validate_open_handle(handle) -> None:
        if handle.attrs.get("schema") != ArtifactStore.RESULT_SCHEMA:
            raise ArtifactError("not an ANYfem result artifact")
        if int(handle.attrs.get("version", 0)) > ArtifactStore.VERSION:
            raise ArtifactError("result artifact is newer than this ANYfem")
        if not bool(handle.attrs.get("complete", False)):
            raise ArtifactError("result artifact is incomplete")


_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _sha256_bytes(contents: bytes) -> str:
    return "sha256:" + hashlib.sha256(contents).hexdigest()


def _validated_sha256(value: object, context: str) -> str:
    checksum = str(value)
    if not _SHA256_PATTERN.fullmatch(checksum):
        raise ArtifactError(
            f"{context} must use the form sha256:<64 lowercase hex digits>"
        )
    return checksum


def _safe_source_name(value: object) -> str:
    name = str(value)
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ArtifactError(
            "embedded source name must be a plain file name without a path"
        )
    return name


def _source_format(value: object) -> str:
    source_format = str(value)
    if not source_format or "\x00" in source_format:
        raise ArtifactError("embedded source format must be a non-empty string")
    return source_format


def _prepare_imported_model(
    imported_model: Mapping[str, Any] | None,
    embedded_source: bytes | bytearray | memoryview | None,
) -> tuple[dict[str, Any] | None, bytes | None]:
    if imported_model is None:
        if embedded_source is not None:
            raise ArtifactError(
                "embedded source bytes require imported-model metadata"
            )
        return None, None
    try:
        payload = json.loads(_json(imported_model))
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"imported-model metadata is not JSON data: {error}") from None
    if not isinstance(payload, dict):
        raise ArtifactError("imported-model metadata must be a JSON object")
    if embedded_source is None:
        return payload, None
    if not isinstance(embedded_source, (bytes, bytearray, memoryview)):
        raise ArtifactError("embedded source must be supplied as bytes")
    contents = bytes(embedded_source)
    if not contents:
        raise ArtifactError("embedded source cannot be empty")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ArtifactError(
            "embedded source bytes require imported_model.source metadata"
        )
    source_metadata = dict(source)
    source_metadata["name"] = _safe_source_name(source_metadata.get("name", ""))
    source_metadata["format"] = _source_format(
        source_metadata.get("format", payload.get("format", ""))
    )
    payload_format = payload.get("format")
    if (
        payload_format is not None
        and str(payload_format) != source_metadata["format"]
    ):
        raise ArtifactError(
            "imported-model format disagrees with embedded source format"
        )
    actual_checksum = _sha256_bytes(contents)
    expected_checksum = source_metadata.get("sha256")
    if expected_checksum is not None:
        expected_checksum = _validated_sha256(
            expected_checksum, "embedded source checksum"
        )
        if expected_checksum != actual_checksum:
            raise ArtifactError(
                "embedded source checksum does not match its contents"
            )
    source_metadata["sha256"] = actual_checksum
    expected_size = source_metadata.get("byte_size")
    if expected_size is not None and int(expected_size) != len(contents):
        raise ArtifactError(
            "embedded source byte_size does not match its contents"
        )
    source_metadata["byte_size"] = len(contents)
    payload["source"] = source_metadata
    payload.setdefault("format", source_metadata["format"])
    return payload, contents


def _text_attribute(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_json_dataset(group, name: str) -> Any:
    raw = group[name][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _read_embedded_source(
    imported_group, payload: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ArtifactError(
            "embedded source is missing its source metadata object"
        )
    name = _safe_source_name(source.get("name", ""))
    source_format = _source_format(
        source.get("format", payload.get("format", ""))
    )
    dataset = imported_group["source_bytes"]
    array = np.asarray(dataset)
    if array.ndim != 1 or array.dtype != np.dtype(np.uint8):
        raise ArtifactError("embedded source dataset must be one-dimensional bytes")
    contents = bytes(array)
    expected_size = int(source.get("byte_size", -1))
    if expected_size != len(contents):
        raise ArtifactError("embedded source byte_size does not match its dataset")
    expected_checksum = _validated_sha256(
        source.get("sha256", ""), "embedded source checksum"
    )
    actual_checksum = _sha256_bytes(contents)
    if actual_checksum != expected_checksum:
        raise ArtifactError("embedded source checksum does not match its dataset")

    # Attributes duplicate the JSON intentionally: a partial/manual HDF5 edit
    # cannot redirect restoration by changing only one representation.
    attribute_name = _safe_source_name(dataset.attrs.get("name", ""))
    attribute_format = _source_format(dataset.attrs.get("format", ""))
    attribute_checksum = _validated_sha256(
        dataset.attrs.get("sha256", ""), "embedded source dataset checksum"
    )
    if (
        attribute_name != name
        or attribute_format != source_format
        or attribute_checksum != expected_checksum
    ):
        raise ArtifactError(
            "embedded source metadata disagrees with its HDF5 dataset"
        )
    return contents, {
        "name": name,
        "format": source_format,
        "sha256": expected_checksum,
        "byte_size": len(contents),
    }


@contextmanager
def _atomic_destination(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        # Reopen after the writer closed to reject truncated/incomplete output.
        h5py = _h5py()
        with h5py.File(temporary, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                raise ArtifactError("refusing to commit an incomplete artifact")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _atomic_plain_destination(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode_log(text: str) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"log line {line_number} is not a JSON object")
        entry = dict(value)
        if not isinstance(entry.get("timestamp"), str):
            raise ValueError(f"log line {line_number} has no timestamp")
        if not isinstance(entry.get("kind"), str):
            raise ValueError(f"log line {line_number} has no event kind")
        if not isinstance(entry.get("message"), str):
            raise ValueError(f"log line {line_number} has no message")
        entries.append(entry)
    return tuple(entries)


def _root_attributes(handle, *, schema: str, artifact_id: str, document_id: str) -> None:
    handle.attrs["schema"] = schema
    handle.attrs["version"] = ArtifactStore.VERSION
    handle.attrs["artifact_id"] = artifact_id
    handle.attrs["document_id"] = document_id
    handle.attrs["created_utc"] = _utc_now()
    handle.attrs["complete"] = False


def _dataset(group, name: str, values: np.ndarray, *, frame_major: bool = False):
    array = np.asarray(values)
    chunks = True
    if array.size == 0:
        chunks = None
    elif frame_major and array.ndim > 1:
        chunks = (1,) + tuple(array.shape[1:])
    return group.create_dataset(
        name,
        data=array,
        compression="gzip" if array.size else None,
        compression_opts=4 if array.size else None,
        shuffle=bool(array.size),
        fletcher32=bool(array.size),
        chunks=chunks,
    )


def _artifact(uri: str, identifier: str, kind: str, path: Path) -> ArtifactRef:
    return ArtifactRef(
        id=identifier,
        kind=kind,
        uri=uri,
        byte_size=path.stat().st_size,
        sha256=_sha256(path),
        created_utc=_utc_now(),
    )

"""Plain-data transport for completed results that refuse to pickle.

Why this exists
---------------
A worker process returns its completed result through a pipe, which means
pickling it.  A linear static result (``LinearSolution``) carries the built
solver model, and a model built with the qualified S3 V2D shell formulation
holds *live* element objects whose pickle support is deliberately refused
(``python_pickle_restart`` is blocked pending a successor gate).  The solve is
complete; only the transfer to the application fails.

The refusing element family instead supports stateless plain-data
serialization (``to_dict`` / ``from_dict``, with the solver's own identity and
schema validation inside ``from_dict``).  This module uses exactly those
supported APIs to move a completed *linear* result across the process
boundary: the worker encodes the result into plain data, the parent decodes it
after the worker tree is confirmed retired.

What it deliberately does not do
--------------------------------
* It does not weaken, bypass or re-implement the solver's pickle refusal; the
  live elements are never pickled and no reducer or monkeypatch is installed.
* It does not rebuild physics: nodes, materials, boundary conditions,
  constraint equations, load cases and the DOF manager travel verbatim; only
  the refusing elements are re-constructed by the solver's own ``from_dict``,
  which validates the formulation fingerprints itself.
* It does not carry nonlinear accepted state or history.  The element
  serialization is stateless configuration, so nonlinear results are refused
  with a structured error instead of being silently stripped.  The supported
  scope is the exact linear static classes (``LinearSolution`` and
  ``LinearBatchSolution``); anything else keeps the historical behaviour (a
  structured ``ResultTransferError``).
* It does not transport the solver mesh's private qualified-direct-state
  token.  That token is derived runtime cache bookkeeping, not model state:
  the decoded mesh is rebuilt through the public ``FEMesh`` constructor, which
  owns a fresh mutation epoch and re-subscribes the decoded nodes and elements
  to it.  Transporting the worker's token would leave the reconstructed mesh
  holding an epoch that public mutations do not advance and that nothing is
  subscribed to, silently breaking qualified cache invalidation.

Nothing is dropped silently.  Every dataclass field of a transferred object is
carried (or the transfer fails), the field set is verified on decode, exact
classes are enforced on both sides, and element/node identities must match
their map keys.  The two dynamic attributes the application attaches to a
completed ``LinearSolution`` in the worker (``_requested_global_stress`` and
``_requested_patch_stress``, read back by the result-artifact adapter) are
carried in a dedicated record or the transfer fails; any other dynamic
attribute on the instance fails closed rather than being silently dropped.
A result that cannot be proven identical is refused, never transferred
changed.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

from .job_process import dumps, loads

__all__ = [
    "ResultTransportError",
    "encode_result",
    "decode_result",
]

CODEC_ID = "anyfem.result_transport"
CODEC_VERSION = 3

_ELEMENT_PICKLE = "pickle"
_ELEMENT_DICT = "element_dict"

_STRESS_NONE = "none"
_STRESS_CARRIED = "carried"

#: The two dynamic attributes ``application/workflow.py`` attaches to a
#: completed exact ``LinearSolution`` in the worker before returning it and
#: ``io/result_artifact.py`` reads back by name on the application side.
#: They are not dataclass fields, so a field-only codec would silently drop
#: them; each present attribute is carried explicitly or the transfer fails.
_APPLICATION_EXTRAS = ("_requested_global_stress", "_requested_patch_stress")

#: Sentinel for "no stress cache travelled"; distinct from a carried None.
_NO_CACHE = object()


class ResultTransportError(RuntimeError):
    """A completed result cannot travel through the plain-data transport.

    The message names the concrete boundary (unsupported result type, a field
    with no supported plain-data route, or a decode verification failure) so
    the structured job failure stays actionable.
    """


# ----------------------------------------------------------------------
# worker side: encode
# ----------------------------------------------------------------------
def encode_result(result: Any) -> dict[str, Any]:
    """Encode a completed result into picklable plain data.

    Raises :class:`ResultTransportError` for anything outside the exact
    supported linear static classes, or when a required piece has no supported
    plain-data route.  The caller reports that as the same structured
    result-transfer failure as an unpicklable result.
    """

    from .post.results import LinearBatchSolution, LinearSolution

    supported = (LinearSolution, LinearBatchSolution)
    if type(result) in supported:
        kind = "LinearSolution" if type(result) is LinearSolution else "LinearBatchSolution"
        return {
            "codec": CODEC_ID,
            "version": CODEC_VERSION,
            "kind": kind,
            "record": _encode_dataclass(result, "the completed result"),
        }
    if isinstance(result, supported):
        raise ResultTransportError(
            f"a completed {_class_location(type(result))} is a subclass of the "
            "supported linear static result family; its extra fields cannot be "
            "carried without verification, so the transfer refuses it rather "
            "than silently reducing it to its base class"
        )
    raise ResultTransportError(
        f"a completed {type(result).__name__} has no supported plain-data "
        "transfer; the supported scope is the exact linear static family "
        "(LinearSolution, LinearBatchSolution). Nonlinear accepted state and "
        "history are not carried by the stateless element serialization and "
        "are refused rather than stripped"
    )


def _class_location(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_solver_mesh(value: Any) -> bool:
    from anysolver.fe_core import FEMesh

    return type(value) is FEMesh


def _is_mesh_record(record: Any) -> bool:
    from anysolver.fe_core import FEMesh

    return isinstance(record, dict) and record.get("class") == _class_location(FEMesh)


def _pickle_field(owner: str, name: str, value: Any) -> bytes:
    try:
        return dumps(value)
    except Exception as error:
        raise ResultTransportError(
            f"{owner} field {name!r} cannot be transferred through the "
            f"plain-data transport: {type(error).__name__}: {error}"
        ) from error


def _encode_dataclass(obj: Any, owner: str) -> dict[str, Any]:
    """Encode every dataclass field of ``obj``; nothing is dropped silently.

    Field dispatch is by name: ``built``/``fe_model``/``mesh`` have dedicated
    encoders, ``shapes`` encodes each batch shape, ``_stress`` carries the
    recovered-stress cache exactly or fails, and every other field travels
    through the transport pickler.  A field that cannot travel fails the
    transfer with its name in the message.
    """

    cls = type(obj)
    values: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        name = field.name
        value = getattr(obj, name)
        if name == "built":
            values[name] = _encode_built(value)
        elif name == "fe_model":
            values[name] = _encode_fe_model(value)
        elif name == "mesh" and _is_solver_mesh(value):
            # FEModel.mesh is the solver FEMesh; BuiltModel.mesh is the
            # application mesh, which pickles and travels verbatim.
            values[name] = _encode_mesh(value)
        elif name == "shapes":
            values[name] = [_encode_shape(shape) for shape in value]
        elif name == "elements":
            values[name] = _encode_elements(value)
        elif name == "_stress":
            values[name] = _encode_stress_cache(obj)
        else:
            values[name] = _pickle_field(owner, name, value)
    record = {
        "class": _class_location(cls),
        "fields": [field.name for field in dataclasses.fields(cls)],
        "values": values,
    }
    if _is_linear_solution(cls):
        record["application_extras"] = _encode_application_extras(obj, owner)
    return record


def _is_linear_solution(cls: type) -> bool:
    from .post.results import LinearSolution

    return cls is LinearSolution


def _encode_application_extras(solution: Any, owner: str) -> dict[str, bytes]:
    """Carry the application's two known requested-stress caches, or fail.

    The workflow attaches ``_requested_global_stress`` and
    ``_requested_patch_stress`` to a completed ``LinearSolution`` in the
    worker, and the result-artifact adapter reads them back by those exact
    names.  They are not dataclass fields, so they travel in a dedicated
    record: each present attribute is pickled (carried) or the transfer
    fails.  Any other dynamic attribute on the instance fails closed: the
    transport cannot verify undocumented physics state, so it refuses
    rather than silently drop it.
    """

    field_names = {field.name for field in dataclasses.fields(type(solution))}
    instance_attrs = set(vars(solution))
    unexpected = sorted(instance_attrs - field_names - set(_APPLICATION_EXTRAS))
    if unexpected:
        raise ResultTransportError(
            f"{owner} carries unsupported dynamic attribute(s) {unexpected}; "
            "the transfer refuses rather than silently dropping state it "
            "cannot verify"
        )
    encoded: dict[str, bytes] = {}
    for name in _APPLICATION_EXTRAS:
        if name in instance_attrs:
            encoded[name] = _pickle_field(owner, name, getattr(solution, name))
    return encoded


def _encode_shape(shape: Any) -> dict[str, Any]:
    from .post.results import LinearSolution

    if type(shape) is not LinearSolution:
        raise ResultTransportError(
            f"a linear batch contains a shape of type {type(shape).__name__}, "
            "which is outside the exact LinearSolution family; the transfer "
            "refuses it rather than transferring it with unverified fields"
        )
    return _encode_dataclass(shape, "a batch shape")


def _encode_stress_cache(solution: Any) -> dict[str, Any]:
    """The recovered-stress cache: carried exactly, or the transfer fails.

    The cache is provenance-bearing recovered state, not disposable derived
    data, so an unpicklable cache is a structured transfer failure -- never a
    silent drop that would quietly change what later stress reads return.
    """

    stress = getattr(solution, "_stress", None)
    if stress is None:
        return {"state": _STRESS_NONE}
    return {
        "state": _STRESS_CARRIED,
        "blob": _pickle_field("the completed result", "_stress", stress),
    }


def _encode_built(built: Any) -> dict[str, Any]:
    from .solve.build import BuiltModel

    if type(built) is not BuiltModel:
        raise ResultTransportError(
            f"the completed result carries a built model of type "
            f"{_class_location(type(built))}; only the exact BuiltModel "
            "dataclass is supported, and anything else is refused rather "
            "than transferred with unverified fields"
        )
    return _encode_dataclass(built, "the built model")


def _encode_fe_model(fe_model: Any) -> dict[str, Any]:
    from anysolver.fe_core import FEModel

    if type(fe_model) is not FEModel:
        raise ResultTransportError(
            f"the completed result carries an FE model of type "
            f"{_class_location(type(fe_model))}; only the exact solver FEModel "
            "is supported, and anything else is refused rather than "
            "transferred with unverified fields"
        )
    return _encode_dataclass(fe_model, "the FE model")


def _encode_mesh(mesh: Any) -> dict[str, Any]:
    from anysolver.fe_core import FEMesh

    if type(mesh) is not FEMesh:
        raise ResultTransportError(
            f"the completed result carries an FE mesh of type "
            f"{_class_location(type(mesh))}; only the exact solver FEMesh is "
            "supported, and anything else is refused rather than transferred "
            "with unverified fields"
        )
    # The mesh's private qualified-direct-state token is derived runtime
    # bookkeeping, not transferred state: the public FEMesh constructor on
    # decode owns a fresh mutation epoch and re-subscribes the decoded nodes
    # and elements to it.  Only the public dataclass fields travel.
    return _encode_dataclass(mesh, "the FE mesh")


def _encode_elements(elements: Any) -> dict[int, dict[str, Any]]:
    """The elements, each through its own supported route.

    An element that pickles travels as itself.  An element that refuses pickle
    travels through its own ``to_dict`` plain-data API with its class location
    recorded for dispatch; the solver's ``from_dict`` re-validates the identity
    fingerprints on decode.  An element that offers neither route fails the
    transfer.
    """

    encoded: dict[int, dict[str, Any]] = {}
    for key, element in elements.items():
        element_id = int(key)
        try:
            blob = dumps(element)
        except Exception:  # noqa: BLE001 - the refusal is the supported signal
            to_dict = getattr(element, "to_dict", None)
            from_dict = getattr(type(element), "from_dict", None)
            if not callable(to_dict) or not callable(from_dict):
                raise ResultTransportError(
                    f"element {element_id} ({type(element).__name__}) refuses "
                    "pickle and offers no supported to_dict/from_dict "
                    "plain-data API, so the completed result cannot be "
                    "transferred"
                ) from None
            encoded[element_id] = {
                "codec": _ELEMENT_DICT,
                "module": type(element).__module__,
                "qualname": type(element).__qualname__,
                "payload": to_dict(),
            }
        else:
            encoded[element_id] = {"codec": _ELEMENT_PICKLE, "blob": blob}
    return encoded


# ----------------------------------------------------------------------
# application side: decode
# ----------------------------------------------------------------------
def decode_result(envelope: dict[str, Any]) -> Any:
    """Rebuild a completed result from its plain-data envelope.

    Called in the waiting job thread after the worker tree is confirmed
    retired and before the result is returned/attached to the job record.
    Every verification below fails closed: a result that cannot be proven
    identical in structure is a transfer failure, never a silently different
    model.
    """

    if not isinstance(envelope, dict):
        raise ResultTransportError("the encoded result is not a mapping")
    if envelope.get("codec") != CODEC_ID or envelope.get("version") != CODEC_VERSION:
        raise ResultTransportError(
            f"unknown result transport codec {envelope.get('codec')!r} "
            f"(version {envelope.get('version')!r})"
        )
    kind = envelope.get("kind")
    if kind == "LinearSolution":
        from .post.results import LinearSolution

        return _decode_dataclass(envelope["record"], LinearSolution, "the completed result")
    if kind == "LinearBatchSolution":
        from .post.results import LinearBatchSolution

        return _decode_dataclass(
            envelope["record"], LinearBatchSolution, "the completed result"
        )
    raise ResultTransportError(f"unknown encoded result kind {kind!r}")


def _decode_dataclass(record: Any, cls: type, owner: str) -> Any:
    """Rebuild ``cls`` from its encoded record, verifying the field set.

    The encoded field names must exactly match the current class's fields, so
    a codec/class drift (or a tampered envelope) fails closed instead of
    guessing.  Init fields go through the supported constructor; init=False
    fields are restored explicitly and verified, so derived identity (the
    shell formulation policy) must survive identically or the transfer fails.
    """

    if not isinstance(record, dict):
        raise ResultTransportError(f"{owner} envelope is not a mapping")
    if record.get("class") != _class_location(cls):
        raise ResultTransportError(
            f"{owner} was encoded as {record.get('class')!r}, not the current "
            f"{_class_location(cls)!r}"
        )
    names = [field.name for field in dataclasses.fields(cls)]
    if record.get("fields") != names:
        raise ResultTransportError(
            f"{owner} was encoded with the field set {record.get('fields')!r}, "
            f"which does not match the current fields {names!r}; the transfer "
            "refuses rather than guess which fields changed"
        )
    values = record.get("values")
    if not isinstance(values, dict) or set(values) != set(names):
        raise ResultTransportError(
            f"{owner} envelope values do not cover its fields exactly"
        )
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        name = field.name
        if name == "built":
            kwargs[name] = _decode_built(values[name])
        elif name == "fe_model":
            kwargs[name] = _decode_fe_model(values[name])
        elif name == "mesh" and _is_mesh_record(values[name]):
            kwargs[name] = _decode_mesh(values[name])
        elif name == "shapes":
            kwargs[name] = [_decode_shape(shape) for shape in values[name]]
        elif name == "elements":
            kwargs[name] = _decode_elements(values[name])
        elif name == "_stress":
            stress = _decode_stress_cache(values[name])
            if stress is not _NO_CACHE:
                kwargs[name] = stress
        else:
            kwargs[name] = _loads_field(owner, name, values[name])
    obj = cls(**kwargs)
    for field in dataclasses.fields(cls):
        if field.init:
            continue
        expected = _loads_field(owner, field.name, values[field.name])
        if getattr(obj, field.name) != expected:
            raise ResultTransportError(
                f"{owner} field {field.name!r} did not survive the transfer "
                "identically; the result is refused rather than transferred "
                "changed"
            )
    if _is_linear_solution(cls):
        _decode_application_extras(record, obj, owner)
    elif "application_extras" in record:
        raise ResultTransportError(
            f"{owner} envelope carries application extras, which only the "
            "exact LinearSolution record may hold"
        )
    return obj


def _decode_application_extras(record: dict[str, Any], solution: Any, owner: str) -> None:
    """Restore the carried requested-stress caches, presence exactly as encoded.

    A LinearSolution record must carry the application-extras record (an
    absent record means codec drift, and the transfer fails closed).  Only
    the two known names are accepted; each present name is restored by
    ``setattr`` with its decoded value, and an absent name stays absent, so
    presence versus absence survives the transfer exactly.
    """

    encoded = record.get("application_extras")
    if not isinstance(encoded, dict):
        raise ResultTransportError(
            f"{owner} envelope has no application-extras record; the transfer "
            "refuses rather than guess which requested stress caches travelled"
        )
    unexpected = sorted(set(encoded) - set(_APPLICATION_EXTRAS))
    if unexpected:
        raise ResultTransportError(
            f"{owner} envelope carries unknown application extras {unexpected}"
        )
    for name in _APPLICATION_EXTRAS:
        if name in encoded:
            setattr(solution, name, _loads_field(owner, name, encoded[name]))


def _loads_field(owner: str, name: str, blob: bytes) -> Any:
    try:
        return loads(blob)
    except Exception as error:
        raise ResultTransportError(
            f"{owner} field {name!r} could not be decoded: "
            f"{type(error).__name__}: {error}"
        ) from error


def _decode_shape(record: Any) -> Any:
    from .post.results import LinearSolution

    return _decode_dataclass(record, LinearSolution, "a batch shape")


def _decode_stress_cache(encoded: Any) -> Any:
    if not isinstance(encoded, dict):
        raise ResultTransportError("the stress cache envelope is not a mapping")
    state = encoded.get("state")
    if state == _STRESS_NONE:
        return _NO_CACHE
    if state == _STRESS_CARRIED:
        if "blob" not in encoded:
            raise ResultTransportError(
                "the stress cache says it was carried but holds no payload"
            )
        return _loads_field("the completed result", "_stress", encoded["blob"])
    raise ResultTransportError(f"unknown stress cache state {state!r}")


def _decode_built(record: Any) -> Any:
    from .solve.build import BuiltModel

    return _decode_dataclass(record, BuiltModel, "the built model")


def _decode_fe_model(record: Any) -> Any:
    from anysolver.fe_core import FEModel

    fe_model = _decode_dataclass(record, FEModel, "the FE model")
    _verify_fe_model(fe_model)
    return fe_model


def _decode_mesh(record: Any) -> Any:
    from anysolver.fe_core import FEMesh

    # The public FEMesh constructor called below owns a fresh runtime cache
    # epoch for the reconstructed mesh and re-subscribes the decoded nodes and
    # elements to it.  The worker's private qualified-direct-state token is
    # deliberately neither transported nor overwritten: a transported epoch
    # would not be advanced by public mutations of the decoded mesh and would
    # silently break qualified cache invalidation.
    return _decode_dataclass(record, FEMesh, "the FE mesh")


def _decode_elements(encoded: Any) -> dict[int, Any]:
    elements: dict[int, Any] = {}
    for raw_id, record in encoded.items():
        element_id = int(raw_id)
        if not isinstance(record, dict):
            raise ResultTransportError(
                f"element {element_id} has no transport record"
            )
        codec = record.get("codec")
        if codec == _ELEMENT_PICKLE:
            if "blob" not in record:
                raise ResultTransportError(
                    f"element {element_id} travelled as itself but holds no blob"
                )
            element = _loads_field("the FE mesh", f"element {element_id}", record["blob"])
        elif codec == _ELEMENT_DICT:
            element = _decode_dict_element(element_id, record)
        else:
            raise ResultTransportError(
                f"element {element_id} has unknown transport codec {codec!r}"
            )
        identity = getattr(element, "element_id", None)
        if identity is None:
            identity = getattr(element, "id", None)
        if identity is None or int(identity) != element_id:
            raise ResultTransportError(
                f"element stored under id {element_id} carries identity "
                f"{identity!r}; the transfer refuses a model whose element "
                "ids do not match their map keys"
            )
        elements[element_id] = element
    return elements


def _decode_dict_element(element_id: int, record: dict[str, Any]) -> Any:
    """One refusing element, rebuilt by its own exact class ``from_dict``.

    The class is located by the recorded module/qualname and the payload is
    handed to the solver's own ``from_dict``, which validates the exact
    class, the schema keys and every identity fingerprint.  The decoded
    element must be exactly the advertised class and must re-encode to the
    identical payload, which proves the configuration survived the round trip
    exactly.
    """

    try:
        module = importlib.import_module(str(record["module"]))
        cls = module
        for part in str(record["qualname"]).split("."):
            cls = getattr(cls, part)
    except (ImportError, AttributeError, KeyError) as error:
        raise ResultTransportError(
            f"element {element_id}'s class {record.get('qualname')!r} from "
            f"module {record.get('module')!r} cannot be loaded: {error}"
        ) from error
    from_dict = getattr(cls, "from_dict", None)
    if not callable(from_dict):
        raise ResultTransportError(
            f"element {element_id}'s class {record.get('qualname')!r} has no "
            "supported from_dict plain-data API"
        )
    try:
        element = from_dict(record["payload"])
    except Exception as error:
        raise ResultTransportError(
            f"element {element_id} refused its encoded payload: "
            f"{type(error).__name__}: {error}"
        ) from error
    if type(element) is not cls:
        raise ResultTransportError(
            f"element {element_id} decoded to {_class_location(type(element))}, "
            f"not the advertised exact class {_class_location(cls)}"
        )
    to_dict = getattr(element, "to_dict", None)
    if not callable(to_dict) or to_dict() != record["payload"]:
        raise ResultTransportError(
            f"element {element_id} did not survive the plain-data round trip "
            "identically; the result is refused rather than transferred changed"
        )
    return element


def _verify_fe_model(fe_model: Any) -> None:
    """Fail closed unless the decoded model is structurally the encoded one.

    The DOF manager and nodes travelled verbatim, so any mismatch here means
    the envelope was damaged in transit; that is a transfer failure, never a
    quietly remapped model (the displacement vector is indexed by these DOFs).
    """

    mesh = fe_model.mesh
    manager = mesh.dof_manager
    seen_dofs: set[int] = set()
    for node_id, node in mesh.nodes.items():
        if int(getattr(node, "id", -1)) != int(node_id):
            raise ResultTransportError(
                f"node stored under id {node_id} carries id "
                f"{getattr(node, 'id', None)!r}; the transfer refuses a model "
                "whose node ids do not match their map keys"
            )
        dofs = list(node.dofs)
        if dofs != list(manager.get_node_dofs(int(node_id))):
            raise ResultTransportError(
                f"node {node_id}'s DOF mapping did not survive the transfer"
            )
        if len(dofs) != len(set(dofs)) or any(dof in seen_dofs for dof in dofs):
            raise ResultTransportError(
                f"node {node_id} carries a duplicated or shared DOF"
            )
        seen_dofs.update(dofs)
    if seen_dofs != set(range(manager.total_dofs)):
        raise ResultTransportError(
            "the decoded DOF mapping does not cover the model's DOFs exactly"
        )
    for element_id, element in mesh.elements.items():
        node_ids = getattr(element, "node_ids", ())
        missing = [int(n) for n in node_ids if int(n) not in mesh.nodes]
        if missing:
            raise ResultTransportError(
                f"element {element_id} references missing nodes {missing}"
            )
        material_name = getattr(element, "material_name", None)
        if material_name is not None and material_name not in fe_model.materials:
            raise ResultTransportError(
                f"element {element_id} references missing material "
                f"{material_name!r}"
            )

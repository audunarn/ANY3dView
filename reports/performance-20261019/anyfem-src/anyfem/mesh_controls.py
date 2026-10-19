"""Toolkit-neutral, serializable controls for the public mesher API."""

from dataclasses import asdict, dataclass


def native_v2_options_type():
    """Older supported mesher generations keep legacy GUI controls usable."""
    try:
        from anymesher.native_v2 import NativeMeshingOptions
    except ImportError:
        return None
    return NativeMeshingOptions


@dataclass(frozen=True)
class MeshControls:
    recombine: bool = True
    certification_mode: str = "interactive"
    point_placement: str = "legacy_lattice"
    metric_mode: str = "legacy"
    max_insertions: int = 10_000
    max_topology_operations: int = 1_000_000
    cancellation_interval: int = 256

    def __post_init__(self):
        from anymesher.hybrid import CertificationMode

        if type(self.recombine) is not bool:
            raise ValueError("quad recombination must be Boolean")
        if self.certification_mode not in {"interactive", "strict"}:
            raise ValueError("certification must be interactive or strict")
        CertificationMode(self.certification_mode)
        self.native_options()  # Owner validation is authoritative.

    def native_options(self):
        values = {
            key: value for key, value in asdict(self).items()
            if key not in {"recombine", "certification_mode"}
        }
        options_type = native_v2_options_type()
        if options_type is None:
            defaults = {key: type(self).__dataclass_fields__[key].default for key in values}
            if values != defaults:
                raise ValueError("Installed ANYmesher does not provide native-v2 controls; use legacy defaults")
            return None
        return options_type(**values)

    def parameters(self):
        return {
            ("recombine" if key == "recombine" else "native_" + key): value
            for key, value in asdict(self).items() if key != "certification_mode"
        }

    def effective_dict(self, strategy):
        # Mapped meshing never invokes a native triangulator/filler.
        return ({"certification_mode": self.certification_mode}
                if strategy == "mapped" else asdict(self))

    @classmethod
    def display_values(cls, settings):
        """Read saved intent for display even if execution capability is absent."""
        values = {key: field.default for key, field in cls.__dataclass_fields__.items()}
        if settings is None:
            return values
        parameters = dict(settings.parameters)
        values.update({
            key: parameters["native_" + key]
            for key in cls.__dataclass_fields__
            if "native_" + key in parameters
        })
        values["recombine"] = parameters.get("recombine", True)
        values["certification_mode"] = str(settings.certification_mode.value)
        return values

    @classmethod
    def from_settings(cls, settings):
        return cls(**cls.display_values(settings))


@dataclass(frozen=True)
class StructuredMeshControls:
    """Serializable GUI controls for ANYmesher's global structure planner.

    Preference and element-quality limits remain first-class Mesh-panel
    fields.  This object carries the remaining public
    ``StructuredMeshingOptions`` values so background jobs, previews and saved
    projects all use the same explicit policy.
    """

    allow_detached_partition: bool = True
    max_element_growth: float = 1.5
    minimum_size_ratio: float = 0.67
    maximum_size_ratio: float = 1.5
    maximum_radial_sides: int = 8
    maximum_candidates_per_component: int = 256
    maximum_face_records: int = 100_000
    maximum_blocks: int = 100_000
    maximum_edge_records: int = 200_000
    maximum_estimated_elements: int = 2_000_000
    maximum_divisions_per_edge: int = 100_000

    def __post_init__(self):
        # The owning package remains authoritative for range and combination
        # validation.  Constructing its public options object also prevents a
        # future ANYmesher field change from being silently accepted here.
        self.owner_options()

    def owner_options(self, *, preference="balanced", quality_policy=None):
        from anymesher.structured import StructuredMeshingOptions

        return StructuredMeshingOptions(
            preference=preference,
            quality_policy={} if quality_policy is None else quality_policy,
            **asdict(self),
        )

    def parameters(self):
        return {
            "structured_" + key: value for key, value in asdict(self).items()
        }

    def effective_dict(self):
        return asdict(self)

    @classmethod
    def display_values(cls, settings):
        values = {
            key: field.default for key, field in cls.__dataclass_fields__.items()
        }
        if settings is not None:
            parameters = dict(settings.parameters)
            values.update(
                {
                    key: parameters["structured_" + key]
                    for key in cls.__dataclass_fields__
                    if "structured_" + key in parameters
                }
            )
        return values

    @classmethod
    def from_settings(cls, settings):
        return cls(**cls.display_values(settings))

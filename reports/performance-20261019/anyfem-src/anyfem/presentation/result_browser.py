"""Metadata-only result navigation; numerical recovery stays with its owners."""
from dataclasses import dataclass

from ..post.fields import DISPLACEMENT_FIELDS, STRESS_FIELDS, field_unit


@dataclass(frozen=True)
class ResultEntry:
    path: tuple[str, ...]
    label: str
    kind: str
    key: str
    component: str | None = None
    unit: str = ""
    location: str = ""

    @property
    def identity(self):
        return self.kind, self.key, self.component


def result_family(key, label=""):
    text=(key+" "+label).lower()
    if key in ("rx", "ry", "rz") or "rotation" in text:return "Rotation"
    if key in DISPLACEMENT_FIELDS or "displacement" in text:return "Displacement"
    if "strain" in text or "peeq" in text:return "Strain"
    if key in STRESS_FIELDS or "stress" in text or "von_mises" in text:return "Stress"
    if "reaction" in text:return "Reactions"
    if "velocity" in text:return "Velocity"
    if "acceleration" in text:return "Acceleration"
    if "force" in text or "moment" in text:return "Forces and moments"
    if "energy" in text:return "Energy"
    return "Other quantities"


def component_label(key):
    names={"magnitude":"Magnitude", "von_mises":"Von Mises", "ux":"X (Ux)",
        "uy":"Y (Uy)", "uz":"Z (Uz)", "rx":"X (Rx)", "ry":"Y (Ry)",
        "rz":"Z (Rz)", "equivalent_plastic_strain":"Equivalent plastic strain"}
    if key in {"xx","yy","zz","xy","xz","yz"}:return key.upper()
    return names.get(key, key.replace("_", " ").replace(".", " · "))


def result_entries(*, live_fields=(), live_quantities=(), dataset=None):
    """Use only advertised field names/descriptors; never read field arrays."""
    entries=[]
    for key in live_fields:
        family=result_family(key);path=("Live fields",family);label=component_label(key)
        if family=="Stress":
            prefix=next((name for name in ("membrane","bending","top","bottom","shear")
                         if key.startswith(name+"_")),None)
            # Beam bending/shear have different semantics from shell tensors.
            if key.endswith("stress_y") or key.endswith("stress_z") or key in ("axial_stress","torsional_stress"):
                path+=("Beam",)
            elif prefix:
                path+=(prefix.capitalize()+" surface" if prefix in ("top","bottom") else prefix.capitalize(),)
                label=component_label(key[len(prefix)+1:]).upper()
        entries.append(ResultEntry(path,label,"live_field",key,unit=field_unit(key)))
    for descriptor in live_quantities:
        key=descriptor.quantity_id
        path=("Live quantities",result_family(key,descriptor.label),descriptor.label)
        entries.append(ResultEntry(path[:-1],descriptor.label,"live",key,
                                   unit=descriptor.unit,location=descriptor.location))
        for component in getattr(descriptor,"components",()):
            entries.append(ResultEntry(path,component_label(component),"live",key,component,
                                       descriptor.unit,descriptor.location))
    if dataset is not None:
        for key in dataset.field_keys:
            descriptor=dataset.field(key).descriptor
            family=result_family(getattr(descriptor,"quantity_id",key),descriptor.label)
            path=("Retained fields",family,descriptor.label)
            location=descriptor.location;unit=descriptor.unit
            # Tensor parents inspect the full table; component leaves color
            # exact saved columns. A component norm is never called Von Mises.
            entries.append(ResultEntry(path[:-1],descriptor.label,"field_table",key,
                                       unit=unit,location=location))
            components=tuple(descriptor.components)
            association=f"{key}_{'node' if location=='node' else 'element'}_ids"
            contour=location in {"node","element","element_face","integration_point"} and association in dataset.table_keys
            kind="field" if contour else "field_table"
            if not components and contour:
                entries.append(ResultEntry(path,"Scalar value",kind,key,unit=unit,location=location))
            elif len(components)>1 and contour:
                norm="Translation magnitude" if {"ux","uy","uz"}<=set(components) else "Component norm"
                entries.append(ResultEntry(path,norm,kind,key,unit=unit,location=location))
            for component in components:
                entries.append(ResultEntry(path,component_label(component),kind,key,component,unit,location))
        entries.extend(ResultEntry(("Histories",),key,"history",key) for key in dataset.history_keys)
        entries.extend(ResultEntry(("Tables",),key,"table",key) for key in dataset.table_keys)
    return tuple(entries)

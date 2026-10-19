"""The model tree: everything in the project, and a way to select it.

The tree is a view. Clicking an entity row drives the shared Selection, which
the viewport listens to, so the tree and the 3D view can never disagree about
what is selected.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Iterable, Mapping, Optional
from anygeometry import EntityRef, feature_entity_owners

from ..model.project import Project
from ..model.regions import BooleanRegion, QueryRegion, RegionDomain, RegionStatus
from ..selection import Selection, entity_tag, parse_entity_tag

__all__ = [
    "ModelTree",
    "TREE_ENTITY_ROW_LIMIT",
    "bounded_entity_ids",
    "bounded_unowned_entity_ids",
]


from ..presentation.tree import TREE_ENTITY_ROW_LIMIT, bounded_entity_ids, bounded_unowned_entity_ids

def _vector_text(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):g}" for value in values) + ")"


def _support_values(constraints) -> str:
    parts = []
    for dof in ("ux", "uy", "uz", "rx", "ry", "rz"):
        if dof not in constraints:
            continue
        value = float(constraints[dof])
        unit = "mm" if dof.startswith("u") else "mrad"
        parts.append(f"{dof}={value * 1000.0:g} {unit}")
    return ", ".join(parts)




class ModelTree(ttk.Frame):
    """A collapsible view of materials, sections, geometry and attributes."""

    def __init__(
        self,
        master: tk.Misc,
        project: Project,
        selection: Selection,
        *,
        job_is_stale: Optional[Callable[[object], bool]] = None,
        mesh_is_stale: Optional[Callable[[object], bool]] = None,
    ) -> None:
        super().__init__(master)
        self.project = project
        self.selection = selection
        self._syncing = False
        self._open_state: Dict[str, bool] = {}
        self._row_refs: Dict[str, EntityRef] = {}
        self._region_candidate_cache: dict[str, tuple[EntityRef, ...]] = {}
        self._scroll_fraction = 0.0
        self._action_handler: Optional[
            Callable[[str, tuple[str, ...]], None]
        ] = None
        self._filter_text = tk.StringVar(value="")
        # Generated topology is an implementation detail of its user-authored
        # feature until the user explicitly asks to inspect it.  This state is
        # presentation-only: exploding a row never mutates geometry, changes
        # persistent IDs, or makes a mesh stale.
        self._exploded_feature_ids: set[int] = set()
        self._entity_owner_cache: dict[EntityRef, int] = {}
        self._entity_owner_cache_key: tuple[int, int, int] | None = None
        self._job_is_stale = job_is_stale or (lambda _job: False)
        self._mesh_is_stale = mesh_is_stale or (lambda _mesh: False)

        search = ttk.Frame(self, padding=(4, 4))
        search.pack(fill="x")
        ttk.Label(search, text="Model").pack(side="left")
        entry = ttk.Entry(search, textvariable=self._filter_text, width=18)
        entry.pack(side="right", fill="x", expand=True, padx=(6, 0))
        self._filter_text.trace_add("write", lambda *_args: self.refresh())

        self.tree = ttk.Treeview(self, show="tree", selectmode="extended")
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", lambda _event: self._invoke("edit"))
        self.tree.bind("<Button-3>", self._context_menu)
        self.tree.bind("<Delete>", self._delete_key)
        self.selection.add_listener(self.sync_from_selection)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the tree from the project, preserving what was expanded."""

        self._remember_open_state()
        self._syncing = True
        try:
            self._row_refs.clear()
            self.tree.delete(*self.tree.get_children())
            geometry = self.project.geometry

            settings = self._group("settings", "Project Settings")
            profile = self.project.units
            self._leaf(
                settings,
                "unit:profile",
                "Units   "
                f"{profile.name}   "
                f"[{profile.symbol('length')} / {profile.symbol('force')} / "
                f"{profile.symbol('pressure')}]",
            )

            feature_records = tuple(
                getattr(
                    getattr(geometry, "features", None), "records", ()
                )
            )
            active_feature_ids = {
                int(feature.feature_id) for feature in feature_records
            }
            self._exploded_feature_ids.intersection_update(active_feature_ids)
            entity_owners = self.generated_entity_owners()
            topology_count_by_feature: dict[int, dict[str, int]] = {
                int(feature.feature_id): {
                    "vertex": 0,
                    "edge": 0,
                    "face": 0,
                }
                for feature in feature_records
            }
            owned_by_feature: dict[int, dict[str, list[int]]] = {
                feature_id: {"vertex": [], "edge": [], "face": []}
                for feature_id in self._exploded_feature_ids
            }
            for reference, feature_id in entity_owners.items():
                topology_count_by_feature.setdefault(
                    feature_id,
                    {"vertex": 0, "edge": 0, "face": 0},
                )[reference.kind] += 1
                if feature_id in self._exploded_feature_ids:
                    owned_by_feature.setdefault(
                        feature_id,
                        {"vertex": [], "edge": [], "face": []},
                    )[reference.kind].append(reference.id)
            for feature_entities in owned_by_feature.values():
                for identifiers in feature_entities.values():
                    identifiers.sort()
            features = self._group(
                "features",
                f"Geometry / Features ({len(feature_records)})",
            )
            for feature in feature_records:
                status = "suppressed" if feature.suppressed else feature.state
                feature_entities = owned_by_feature.get(
                    int(feature.feature_id), {}
                )
                feature_counts = (
                    (
                        "plates",
                        topology_count_by_feature[int(feature.feature_id)]["face"],
                    ),
                    (
                        "lines",
                        topology_count_by_feature[int(feature.feature_id)]["edge"],
                    ),
                    (
                        "points",
                        topology_count_by_feature[int(feature.feature_id)]["vertex"],
                    ),
                )
                topology_summary = ", ".join(
                    f"{count} {label}"
                    for label, count in feature_counts
                    if count
                )
                exposure = (
                    "exploded"
                    if int(feature.feature_id) in self._exploded_feature_ids
                    else "internal"
                )
                suffix = (
                    f"   {topology_summary} [{exposure}]"
                    if topology_summary
                    else ""
                )
                feature_row = self._leaf(
                    features,
                    f"feature:{feature.feature_id}",
                    f"{feature.name}   [{status}]{suffix}",
                )
                if int(feature.feature_id) not in self._exploded_feature_ids:
                    continue
                for kind, plural in (
                    ("vertex", "Generated Points"),
                    ("edge", "Generated Lines"),
                    ("face", "Generated Plates"),
                ):
                    identifiers = feature_entities.get(kind, ())
                    if not identifiers:
                        continue
                    branch = self._leaf(
                        feature_row,
                        f"generated_{feature.feature_id}_{kind}",
                        f"{plural} ({len(identifiers)})",
                    )
                    for identifier in bounded_entity_ids(
                        identifiers,
                        self._filter_text.get(),
                        limit=TREE_ENTITY_ROW_LIMIT,
                    ):
                        self._insert_geometry_entity(
                            branch, EntityRef(kind, identifier), generated=True
                        )

            coordinates = self._group(
                "coordinates",
                f"Coordinate Systems ({len(self.project.coordinate_systems)})",
            )
            for identifier, system in sorted(self.project.coordinate_systems.items()):
                self._leaf(
                    coordinates,
                    f"coordinate:{identifier}",
                    f"{system.name}   {system.kind}",
                )

            named_regions = tuple(
                region for region in self.project.regions if not region.hidden
            )
            self._region_candidate_cache: dict[str, tuple[EntityRef, ...]] = {}
            regions = self._group(
                "regions", f"Regions ({len(named_regions)})"
            )
            for region in named_regions:
                status = self._region_status(region)
                definition = region.definition
                definition_label = (
                    definition.operation
                    if isinstance(definition, BooleanRegion)
                    else type(definition).__name__.removesuffix("Region").lower()
                )
                self._leaf(
                    regions,
                    f"region:{region.id}",
                    f"{region.name}   {region.domain.value}/{region.entity_kind}"
                    f"   {definition_label}   [{status.value}]",
                )

            materials = self._group("materials", "Material Definitions")
            for name, material in sorted(self.project.materials.items()):
                if material.symmetry == "isotropic":
                    modulus = material.constants["elastic_modulus"] / 1e9
                    description = f"E={modulus:g} GPa"
                else:
                    description = "orthotropic"
                response = "nonlinear" if material.hardening is not None else "elastic"
                yield_text = f"fy={material.yield_stress / 1e6:g} MPa"
                source = ""
                if material.hardening is not None:
                    hardening = material.hardening
                    if hardening.get("kind") == "dnv_c208":
                        source = (
                            f"   DNV grade={hardening.get('grade', '?')}"
                            f" t={float(hardening.get('thickness', 0.0)) * 1000:g} mm"
                        )
                self._leaf(
                    materials,
                    f"material:{name}",
                    f"{name}   {description}   {yield_text}{source}   [{response}]",
                )

            sections = self._group("sections", "Section Definitions")
            faces_by_section: dict[str, list[int]] = {}
            for identifier, assigned_name in self.project.face_sections.items():
                faces_by_section.setdefault(assigned_name, []).append(identifier)
            edges_by_section: dict[str, list[int]] = {}
            for identifier, assigned_name in self.project.edge_sections.items():
                edges_by_section.setdefault(assigned_name, []).append(identifier)
            for name, plate in sorted(self.project.plate_sections.items()):
                assigned = sorted(faces_by_section.get(name, ()))
                target = (
                    "unassigned"
                    if not assigned
                    else "Model Plate " + ", ".join(map(str, assigned[:8]))
                    + (f" +{len(assigned) - 8}" if len(assigned) > 8 else "")
                )
                self._leaf(
                    sections,
                    f"plate_section:{name}",
                    f"Plate section {name!r}   t={plate.thickness * 1000:g} mm"
                    f"   material={plate.material}   -> {target}",
                )
            for name, beam in sorted(self.project.beam_sections.items()):
                assigned = sorted(edges_by_section.get(name, ()))
                target = (
                    "unassigned"
                    if not assigned
                    else "Model Line " + ", ".join(map(str, assigned[:8]))
                    + (f" +{len(assigned) - 8}" if len(assigned) > 8 else "")
                )
                self._leaf(
                    sections,
                    f"beam_section:{name}",
                    f"Beam section {name!r}   {beam.profile}"
                    f"   material={beam.material}   offset={beam.offset_mode}/"
                    f"{beam.attachment_side}   rotation={beam.rotation_deg:g} deg"
                    f"   -> {target}",
                )

            imperfections = self._group(
                "imperfections",
                f"Geometric Imperfections ({len(self.project.imperfections)})",
            )
            for index, item in enumerate(self.project.imperfections):
                identifier = getattr(item, "id", f"legacy-{index}")
                if item.amplitude is None:
                    amplitude = (
                        "auto (short span / 200)"
                        if item.resolved_kind == "plate_mode"
                        else "auto (length / 300)"
                    )
                else:
                    amplitude = f"{item.amplitude * 1000:g} mm"
                details = (
                    f"waves={item.waves[0]}x{item.waves[1]}"
                    if item.resolved_kind == "plate_mode"
                    else "half-sine bow"
                )
                self._leaf(
                    imperfections,
                    f"imperfection:{identifier}",
                    f"{item.name}   {item.resolved_kind}   {amplitude}   "
                    f"{details} on {item.ref}",
                    ref=item.ref,
                )

            collections = {
                "vertex": geometry.vertices,
                "edge": geometry.edges,
                "face": geometry.faces,
            }
            owned_ids = {
                kind: {
                    reference.id
                    for reference in entity_owners
                    if reference.kind == kind
                }
                for kind in collections
            }
            for kind, branch_key, plural in (
                ("vertex", "points", "User Geometry Points"),
                ("edge", "lines", "User Geometry Lines"),
                ("face", "plates", "User Geometry Plates"),
            ):
                collection = collections[kind]
                generated = owned_ids[kind]
                branch = self._group(
                    branch_key, f"{plural} ({len(collection) - len(generated)})"
                )
                for identifier in bounded_unowned_entity_ids(
                    collection,
                    generated,
                    self._filter_text.get(),
                    limit=TREE_ENTITY_ROW_LIMIT,
                ):
                    self._insert_geometry_entity(
                        branch, EntityRef(kind, identifier), generated=False
                    )

            supports = self._group(
                "supports", f"Supports ({len(self.project.supports)})"
            )
            for index, support in enumerate(self.project.supports):
                dofs = _support_values(support.constraints)
                identifier = getattr(support, "id", f"legacy-{index}")
                self._leaf(
                    supports,
                    f"support:{identifier}",
                    f"{support.name}   {support.ref}   {dofs}",
                    ref=support.ref,
                )

            masses = self._group(
                "masses", f"Masses ({len(self.project.masses)})"
            )
            for index, mass in enumerate(self.project.masses):
                identifier = getattr(mass, "id", f"legacy-{index}")
                self._leaf(
                    masses,
                    f"mass:{identifier}",
                    f"{mass.name}   {mass.value:g} kg at {mass.ref}",
                    ref=mass.ref,
                )

            total_loads = sum(
                len(case.point_loads)
                + len(case.pressures)
                + len(case.line_loads)
                + len(case.surface_tractions)
                + int(case.gravity is not None)
                for case in self.project.load_cases.values()
            )
            loads = self._group("loads", f"Loads ({total_loads})")
            for name, case in sorted(self.project.load_cases.items()):
                case_count = (
                    len(case.point_loads)
                    + len(case.pressures)
                    + len(case.line_loads)
                    + len(case.surface_tractions)
                    + int(case.gravity is not None)
                )
                case_id = getattr(case, "id", name)
                case_node = self._leaf(
                    loads, f"case:{case_id}", f"Case {name} ({case_count})"
                )
                for load in case.point_loads:
                    self._leaf(
                        case_node,
                        f"load:point:{load.id}",
                        f"Point force {_vector_text(load.force)} N; "
                        f"moment {_vector_text(load.moment)} Nm at {load.ref}",
                        ref=load.ref,
                    )
                for load in case.pressures:
                    self._leaf(
                        case_node,
                        f"load:pressure:{load.id}",
                        f"Pressure {load.value:g} Pa on {load.ref}",
                        ref=load.ref,
                    )
                for load in case.line_loads:
                    self._leaf(
                        case_node,
                        f"load:line:{load.id}",
                        f"Line load {_vector_text(load.force_per_length)} N/m on {load.ref}",
                        ref=load.ref,
                    )
                for load in case.surface_tractions:
                    self._leaf(
                        case_node,
                        f"load:traction:{load.id}",
                        f"Surface traction {_vector_text(load.traction)} Pa on {load.ref}",
                        ref=load.ref,
                    )
                if case.gravity is not None:
                    self._leaf(
                        case_node,
                        f"load:{case_id}:gravity",
                        "Gravity / acceleration",
                    )

            meshes = self._group(
                "meshes", f"Meshes ({len(self.project.mesh_records)})"
            )
            for mesh in self.project.mesh_records.values():
                self._leaf(
                    meshes,
                    f"mesh:{mesh.id}",
                    self._mesh_label(mesh),
                )

            output_requests = self._group(
                "output_requests",
                f"Output Requests ({len(self.project.output_requests)})",
            )
            for request in self.project.output_requests.values():
                status = self._output_request_status(request)
                quantities = ", ".join(request.quantity_keys)
                self._leaf(
                    output_requests,
                    f"output_request:{request.id}",
                    f"{request.label}   {quantities} / {request.location}"
                    f"   [{status}]",
                )

            analyses = self._group(
                "analyses", f"Analyses ({len(self.project.analyses)})"
            )
            for analysis in self.project.analyses.values():
                self._leaf(
                    analyses,
                    f"analysis:{analysis.id}",
                    f"{analysis.name}   {analysis.type.replace('_', ' ')}",
                )

            jobs = self._group("jobs", f"Jobs ({len(self.project.jobs)})")
            results = self._group(
                "results",
                "Results "
                f"({sum(job.result_artifact_id is not None for job in self.project.jobs.values())})",
            )
            for job in reversed(tuple(self.project.jobs.values())):
                stale = bool(self._job_is_stale(job))
                self._leaf(jobs, f"job:{job.id}", self._job_label(job, stale))
                if job.result_artifact_id is not None:
                    self._leaf(
                        results,
                        f"result:{job.result_artifact_id}",
                        self._result_label(job, stale),
                    )
        finally:
            self._syncing = False
        self._restore_open_state()
        self.sync_from_selection()

    def _region_status(self, region) -> RegionStatus:
        """Return a cheap, deterministic badge for one reusable scope."""

        if (
            region.domain is RegionDomain.MESH
            and region.mesh_id not in self.project.mesh_records
        ):
            return RegionStatus.STALE
        geometry = self.project.geometry
        candidates: tuple[EntityRef, ...] = ()
        if region.domain is RegionDomain.GEOMETRY:
            collections = {
                "vertex": geometry.vertices,
                "edge": geometry.edges,
                "face": geometry.faces,
            }
            collection = collections.get(region.entity_kind, ())
            if self._region_uses_query(region.id):
                # A query over tens of thousands of owners belongs in the
                # retained/background selection index, not in a tree rebuild.
                # Its AST and dependencies are already validated on insert;
                # report that structural validity without an eager full scan.
                if len(collection) > TREE_ENTITY_ROW_LIMIT:
                    return RegionStatus.VALID
                candidates = self._region_candidate_cache.setdefault(
                    region.entity_kind,
                    tuple(
                        EntityRef(region.entity_kind, int(identifier))
                        for identifier in collection
                    ),
                )
        try:
            return self.project.regions.status(
                region.id,
                geometry=geometry,
                mesh_id=region.mesh_id,
                candidates=candidates,
                feature_resolver=lambda anchor: geometry.features.resolve(
                    anchor, geometry
                ),
            )
        except (AttributeError, KeyError, ValueError):
            return RegionStatus.UNRESOLVED

    def _output_request_status(self, request) -> str:
        """Cheap tree badge without ever fabricating an available field."""

        try:
            region = self.project.regions[request.region.id]
        except (AttributeError, KeyError, ValueError):
            return "unresolved"
        status = self._region_status(region)
        if status is not RegionStatus.VALID:
            return status.value
        if (
            request.basis
            not in ("global", "local", "element", "material", "cylindrical")
            and request.basis not in self.project.coordinate_systems
        ):
            return "unresolved"
        for analysis in self.project.analyses.values():
            if request.id in analysis.output_request_ids and request.problems_for_analysis(
                analysis.type
            ):
                return "invalid"
        return "valid"

    def _region_uses_query(
        self, region_id: str, seen: frozenset[str] = frozenset()
    ) -> bool:
        if region_id in seen:
            return False
        definition = self.project.regions[region_id].definition
        if isinstance(definition, QueryRegion):
            return True
        if not isinstance(definition, BooleanRegion):
            return False
        visited = seen | {region_id}
        return any(
            self._region_uses_query(child, visited)
            for child in definition.region_ids
        )

    def refresh_job_states(self) -> None:
        """Synchronize the small job/result branches without a full rebuild."""

        if not self.tree.exists("jobs") or not self.tree.exists("results"):
            return
        records = tuple(self.project.jobs.values())
        self.tree.item("jobs", text=f"Jobs ({len(records)})")
        self.tree.item(
            "results",
            text="Results "
            f"({sum(job.result_artifact_id is not None for job in records)})",
        )

        for job in reversed(records):
            stale = bool(self._job_is_stale(job))
            job_key = f"job:{job.id}"
            if self.tree.exists(job_key):
                self.tree.item(job_key, text=self._job_label(job, stale))
            else:
                self.tree.insert(
                    "jobs", 0, iid=job_key, text=self._job_label(job, stale)
                )
            if job.result_artifact_id is not None:
                result_key = f"result:{job.result_artifact_id}"
                if self.tree.exists(result_key):
                    self.tree.item(
                        result_key, text=self._result_label(job, stale)
                    )
                else:
                    self.tree.insert(
                        "results",
                        0,
                        iid=result_key,
                        text=self._result_label(job, stale),
                    )
        result_index = 0
        for job_index, job in enumerate(reversed(records)):
            self.tree.move(f"job:{job.id}", "jobs", job_index)
            if job.result_artifact_id is not None:
                self.tree.move(
                    f"result:{job.result_artifact_id}",
                    "results",
                    result_index,
                )
                result_index += 1

    def refresh_mesh_states(self) -> None:
        """Refresh the small mesh branch after a revision without rebuilding."""

        if not self.tree.exists("meshes"):
            return
        records = tuple(self.project.mesh_records.values())
        self.tree.item("meshes", text=f"Meshes ({len(records)})")
        for index, mesh in enumerate(records):
            key = f"mesh:{mesh.id}"
            label = self._mesh_label(mesh)
            if self.tree.exists(key):
                self.tree.item(key, text=label)
            else:
                self.tree.insert("meshes", "end", iid=key, text=label)
            self.tree.move(key, "meshes", index)

    def _mesh_label(self, mesh: object) -> str:
        status = str(getattr(mesh, "status", "") or "completed")
        if self._mesh_is_stale(mesh):
            status = "stale"
        kind = str(getattr(mesh, "kind", "generated"))
        badge = status if status == kind else f"{kind}, {status}"
        return f"{getattr(mesh, 'name', 'Mesh')}   [{badge}]"

    @staticmethod
    def _job_label(job: object, stale: bool) -> str:
        status_value = getattr(job, "status", "unknown")
        status = getattr(status_value, "value", str(status_value))
        if stale:
            status = f"{status}, stale"
        return f"{getattr(job, 'name', 'Job')}   [{status}]"

    @staticmethod
    def _result_label(job: object, stale: bool) -> str:
        label = f"{getattr(job, 'name', 'Job')} results"
        return label + ("   [stale]" if stale else "")

    # ------------------------------------------------------------------
    def _group(self, key: str, label: str) -> str:
        return self.tree.insert("", "end", iid=key, text=label, open=True)

    def _leaf(
        self,
        parent: str,
        key: str,
        label: str,
        *,
        ref: Optional[EntityRef] = None,
    ) -> str:
        item = self.tree.insert(parent, "end", iid=key, text=label)
        if ref is not None:
            self._row_refs[key] = ref
        return item

    def _insert_geometry_entity(
        self,
        parent: str,
        reference: EntityRef,
        *,
        generated: bool,
    ) -> str:
        """Insert one canonical selectable topology row at its intent level."""

        geometry = self.project.geometry
        identifier = int(reference.id)
        origin = "Generated" if generated else "User"
        if reference.kind == "vertex":
            position = geometry.vertex_position(identifier)
            label = (
                f"{origin} Point {identifier}   "
                f"({position[0]:g}, {position[1]:g}, {position[2]:g})"
            )
        elif reference.kind == "edge":
            edge = geometry.edges[identifier]
            curve_kind = type(edge.curve).__name__.lower()
            section = self.project.edge_sections.get(identifier)
            suffix = f"   [{section}]" if section else ""
            label = (
                f"{origin} Line {identifier}   {curve_kind} "
                f"{edge.start}->{edge.end}{suffix}"
            )
        elif reference.kind == "face":
            section = self.project.face_sections.get(identifier)
            suffix = (
                f"   -> section definition {section!r}"
                if section
                else "   -> no section definition assigned"
            )
            label = f"{origin} Plate {identifier}{suffix}"
        else:  # pragma: no cover - guarded by callers and EntityRef.
            raise ValueError(f"unsupported tree geometry kind {reference.kind!r}")
        return self._leaf(parent, entity_tag(reference), label)

    def _visible_ids(self, collection: object) -> list[int]:
        """Return a bounded, filtered set of entity IDs for the virtual tree.

        The viewport owns the full model.  The tree intentionally materialises
        only a useful window for very large models, while a numeric search can
        jump directly to an entity without creating tens of thousands of Tk
        rows.
        """

        query = self._filter_text.get().strip().casefold()
        return bounded_entity_ids(  # type: ignore[arg-type]
            collection,
            query,
            limit=TREE_ENTITY_ROW_LIMIT,
        )

    def set_action_handler(
        self, callback: Optional[Callable[[str, tuple[str, ...]], None]]
    ) -> None:
        """Receive context and double-click actions from the application."""

        self._action_handler = callback

    @property
    def exploded_feature_ids(self) -> frozenset[int]:
        """Feature IDs whose generated topology is exposed in this view."""

        return frozenset(self._exploded_feature_ids)

    def generated_entity_owners(self) -> Mapping[EntityRef, int]:
        """Return the cached exact feature owner of each generated entity."""

        geometry = self.project.geometry
        history = getattr(geometry, "features", None)
        cache_key = (
            id(geometry),
            int(getattr(geometry, "revision", 0)),
            0 if history is None else len(history),
        )
        if cache_key != self._entity_owner_cache_key:
            self._entity_owner_cache = feature_entity_owners(geometry)
            self._entity_owner_cache_key = cache_key
        return self._entity_owner_cache

    def reset_feature_exposure(self) -> None:
        """Collapse generated topology when the tree adopts another project."""

        self._exploded_feature_ids.clear()
        self._entity_owner_cache.clear()
        self._entity_owner_cache_key = None
        self._open_state = {
            key: value
            for key, value in self._open_state.items()
            if not key.startswith("feature:")
            and not key.startswith("generated_")
        }

    def toggle_feature_topology(self, feature_ids: Iterable[int]) -> bool:
        """Explode or collapse feature children without changing the model.

        A mixed selection is exploded as a group.  Selecting only already
        exploded features collapses them.  The return value is the new common
        exposure state.
        """

        identifiers = {
            int(identifier)
            for identifier in feature_ids
            if int(identifier) > 0
        }
        if not identifiers:
            raise ValueError("select at least one geometry feature to explode")
        active = {
            int(record.feature_id)
            for record in getattr(
                getattr(self.project.geometry, "features", None), "records", ()
            )
        }
        missing = identifiers - active
        if missing:
            raise ValueError(
                "geometry feature(s) no longer exist: "
                + ", ".join(map(str, sorted(missing)))
            )
        expose = not identifiers.issubset(self._exploded_feature_ids)
        if expose:
            self._exploded_feature_ids.update(identifiers)
        else:
            self._exploded_feature_ids.difference_update(identifiers)
        self.refresh()
        for identifier in identifiers:
            row = f"feature:{identifier}"
            if self.tree.exists(row):
                self.tree.item(row, open=expose)
            self._open_state[row] = expose
            for kind in ("vertex", "edge", "face"):
                branch = f"generated_{identifier}_{kind}"
                if self.tree.exists(branch):
                    self.tree.item(branch, open=expose)
                self._open_state[branch] = expose
        return expose

    def _context_menu(self, event: tk.Event) -> None:
        row = self.tree.identify_row(event.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        menu = tk.Menu(self, tearoff=False)
        selected_rows = tuple(self.tree.selection())
        selected_count = len(selected_rows)
        selected_features = tuple(
            int(row.split(":", 1)[1])
            for row in selected_rows
            if row.startswith("feature:")
        )
        if selected_features and len(selected_features) == selected_count:
            all_exploded = set(selected_features).issubset(
                self._exploded_feature_ids
            )
            menu.add_command(
                label=(
                    "Collapse generated topology"
                    if all_exploded
                    else "Explode generated topology"
                ),
                command=lambda: self._invoke("explode"),
            )
            menu.add_separator()
        for label, action in (
            ("Edit", "edit"),
            ("Rename", "rename"),
            ("Suppress / Resume", "suppress"),
            ("Duplicate", "duplicate"),
            ("Isolate", "isolate"),
            ("Show Dependencies", "dependencies"),
            ("Delete", "delete"),
        ):
            if action == "delete" and selected_count > 1:
                label = f"Delete {selected_count} selected"
            menu.add_command(label=label, command=lambda value=action: self._invoke(value))
        menu.tk_popup(event.x_root, event.y_root)

    def _invoke(self, action: str) -> None:
        if self._action_handler is not None:
            self._action_handler(action, tuple(self.tree.selection()))

    def _delete_key(self, _event: tk.Event) -> str:
        self._invoke("delete")
        return "break"

    def _remember_open_state(self) -> None:
        view = self.tree.yview()
        if view:
            self._scroll_fraction = float(view[0])
        pending = list(self.tree.get_children())
        while pending:
            key = pending.pop()
            self._open_state[key] = bool(self.tree.item(key, "open"))
            pending.extend(self.tree.get_children(key))

    def _restore_open_state(self) -> None:
        for key, is_open in self._open_state.items():
            if self.tree.exists(key):
                self.tree.item(key, open=is_open)
        self.tree.yview_moveto(self._scroll_fraction)

    # ------------------------------------------------------------------
    def _on_tree_select(self, _event: tk.Event) -> None:
        # Tk fires this while tearing the widget down, when the panels the
        # listeners refresh may already be gone.
        if self._syncing or not self.tree.winfo_exists():
            return
        refs = [
            ref
            for ref in (
                parse_entity_tag(key) or self._row_refs.get(key)
                for key in self.tree.selection()
            )
            if ref is not None
        ]
        if not refs or refs == self.selection.items:
            # Tk delivers <<TreeviewSelect>> asynchronously, so this can arrive
            # after our own sync has already applied it.  Echoing it back would
            # bounce between the tree and the selection forever.
            return
        # Follow the tree into whichever entity mode the user clicked on.
        kinds = {ref.kind for ref in refs}
        if len(kinds) == 1:
            self.selection.set_mode(kinds.pop())
        self.selection.select_many(refs)

    def sync_from_selection(self) -> None:
        """Mirror the shared selection into the tree without echoing back."""

        if self._syncing:
            return
        self._syncing = True
        try:
            keys = [
                tag for tag in self.selection.tags() if self.tree.exists(tag)
            ]
            if list(self.tree.selection()) == keys:
                return
            self.tree.selection_set(keys)
            if keys:
                self.tree.see(keys[0])
        finally:
            self._syncing = False

    def describe_selection(self) -> str:
        return self.selection.describe()

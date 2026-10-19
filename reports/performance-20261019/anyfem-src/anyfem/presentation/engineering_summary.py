"""Engineering interpretation shared by desktop presentations."""
import numpy as np

from .result_summary import nonlinear_path_summary, prescribed_path_progress, submitted_target_load_factor


def material_response(project, analysis):
    names = {project.plate_sections[name].material for name in project.face_sections.values() if name in project.plate_sections}
    plastic = sorted(name for name in names if name in project.materials and project.materials[name].hardening is not None)
    nonlinear = analysis in {"Nonlinear static", "Arc length", "Capacity"}
    if plastic:
        text = ("Shell plasticity active in selected nonlinear analysis: " if nonlinear else "Shell hardening configured (not used by selected analysis): ") + ", ".join(plastic)
        elastic = sorted(names - set(plastic))
        if elastic: text += "; elastic shell materials: " + ", ".join(elastic)
        color = "#1b5e20"
    elif names:
        text = "Shell response is elastic" + (" (analysis is geometrically nonlinear only)" if nonlinear else "") + ": " + ", ".join(sorted(names))
        color = "#b23a00" if nonlinear else "#555555"
    else:
        text, color = "No assigned shell section/material", "#b00020"
    if nonlinear and any(abs(float(value)) > 0 for support in project.supports for value in support.constraints.values()) and not project.imperfections:
        text += "\nBuckling-path warning: the model is geometrically perfect. Add an out-of-plane imperfection; a symmetric flat model can remain on the flat equilibrium path."
        color = "#b23a00"
    return text, color


def constitutive_summary(shape):
    from ..post.results import ImportedSolution
    if isinstance(shape, ImportedSolution):
        return "Constitutive response: unavailable from imported result; attached materials do not establish the producer's constitutive behavior"
    model = shape.built.fe_model
    names = {element.material_name for element in model.mesh.elements.values() if hasattr(element, "thickness")}
    plastic = sorted(name for name in names if getattr(model.get_material(name), "hardening_curve", None) is not None)
    if not plastic:
        return "Constitutive response: shell materials configured as elastic"
    states = getattr(getattr(shape, "raw_result", None), "element_states", {}) or {}
    maxima = [float(np.max(np.asarray(state["alpha"], dtype=float))) for state in states.values() if isinstance(state, dict) and len(state.get("alpha", ()))]
    if not maxima:
        return f"Constitutive response: shell hardening configured ({', '.join(plastic)}); no retained plastic state evidence"
    return f"Constitutive response: NONLINEAR PLASTICITY ACTIVE in retained states ({', '.join(plastic)}); yielded elements {sum(value > 1e-12 for value in maxima)}/{len(maxima)}, max alpha {max(maxima, default=0):.5g}"


def outcome_text(solution, submitted=None):
    path = nonlinear_path_summary(solution, target_load_factor=submitted_target_load_factor(submitted))
    if path is None:
        return str(getattr(solution, "status", "available")).replace("_", " ").upper(), "#2e7d32"
    factor = lambda value: "—" if value is None else f"λ = {value:.5g}"
    lines = [path.status.replace("_", " ").upper(), "Start: λ = 0 (unloaded)", f"First converged: {factor(path.first_converged_load_factor)}", f"Peak: {factor(path.peak_load_factor)}", f"Last / target: {factor(path.last_converged_load_factor)} / {factor(path.target_load_factor)}", path.stop_reason, f"{path.converged_steps} converged increments; {path.total_iterations} Newton iterations"]
    if path.progress_fraction is not None: lines.append(f"{100 * path.progress_fraction:.1f}% of target")
    if path.status.casefold() == "stopped_at_limit":
        lines.append("Numerical path stop; not by itself a verified capacity point")
        if not getattr(getattr(getattr(solution, "built", None), "project", None), "imperfections", ()):
            lines.append("Geometrically perfect model: check imperfection sensitivity before interpreting capacity")
    if path.first_failed_load_factor is not None: lines.append(f"First failed trial λ={path.first_failed_load_factor:.5g}")
    if path.failed_iteration_reason: lines.append(path.failed_iteration_reason)
    if path.max_peeq is not None: lines.append(f"Max PEEQ={path.max_peeq:.5g}")
    lines.extend(prescribed_path_progress(submitted, last_load_factor=path.last_converged_load_factor, target_load_factor=path.target_load_factor))
    return "\n".join(lines), {"success": "#2e7d32", "warning": "#b26a00", "error": "#b00020"}[path.severity]

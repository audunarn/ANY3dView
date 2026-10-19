"""Solver adapter: build an anysolver FEModel and dispatch analyses."""

from .build import BuiltModel, build_fe_model
from .policy import history_modes, recovery_policy, resource_policy
from .prepared_static import (
    PreparedStaticEvaluator,
    PreparedStaticUnavailableError,
    prepare_linear_static_evaluator,
)
from .static_batch import StaticLoadRequest, iter_linear_static_batches
from .prepared_static_response import (
    PeriodicLoadSource, PeriodicSample, PatternLoadRequest,
    PreparedStaticResponseEvaluator, prepare_linear_static_response_evaluator,
)
from .run import (
    ContactConfigurationError,
    eigenmode_imperfection,
    preflight,
    solve_arc_length,
    solve_buckling,
    solve_capacity,
    solve_impact,
    solve_linear_static,
    solve_linear_static_many,
    solve_modal,
    solve_nonlinear_static,
    solve_transient,
)

__all__ = [
    "BuiltModel",
    "StaticLoadRequest",
    "iter_linear_static_batches",
    "PreparedStaticEvaluator",
    "PreparedStaticUnavailableError",
    "prepare_linear_static_evaluator",
    "PeriodicLoadSource",
    "PeriodicSample",
    "PatternLoadRequest",
    "PreparedStaticResponseEvaluator",
    "prepare_linear_static_response_evaluator",
    "ContactConfigurationError",
    "build_fe_model",
    "eigenmode_imperfection",
    "preflight",
    "solve_arc_length",
    "history_modes",
    "recovery_policy",
    "resource_policy",
    "solve_buckling",
    "solve_capacity",
    "solve_impact",
    "solve_linear_static",
    "solve_linear_static_many",
    "solve_modal",
    "solve_nonlinear_static",
    "solve_transient",
]

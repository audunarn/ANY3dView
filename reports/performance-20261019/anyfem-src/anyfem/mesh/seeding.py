"""Compatibility exports for edge seeding now owned by ANYmesher."""

from anymesher.seeding import (
    Seeding,
    SeedingConflict,
    edge_demand,
    edge_distribution,
    solve_seeding,
)

__all__ = [
    "Seeding",
    "SeedingConflict",
    "edge_demand",
    "edge_distribution",
    "solve_seeding",
]

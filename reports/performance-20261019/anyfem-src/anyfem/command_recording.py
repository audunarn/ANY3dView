"""Readable, bounded Python transcripts for GUI command actions."""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from . import commands as command_types
from .commands import Command, CommandEvent, CompositeCommand

__all__ = ["command_event_to_python", "command_to_python"]

_MAX_ITEMS = 40
_MAX_DEPTH = 8
_MAX_LINE = 4000


def _default_value(item) -> tuple[bool, Any]:
    if item.default is not MISSING:
        return True, item.default
    if item.default_factory is not MISSING:
        try:
            return True, item.default_factory()
        except Exception:
            return False, None
    return False, None


def _qualified_type(value: Any) -> tuple[str, bool]:
    kind = type(value)
    if kind is not command_types.AddFeature and issubclass(
        kind, command_types.AddFeature
    ):
        # Generator convenience commands provide a user-friendly custom
        # constructor but inherit AddFeature's dataclass fields.  Serializing
        # those fields with the subclass name produces a non-replayable call
        # such as AddCylinder(kind=..., parameters=...).  The fully populated
        # base feature command is the exact, public replay representation.
        return "commands.AddFeature", True
    if getattr(command_types, kind.__name__, None) is kind:
        return f"commands.{kind.__name__}", True
    module = str(getattr(kind, "__module__", ""))
    qualified = str(getattr(kind, "__qualname__", kind.__name__))
    if (
        module
        and "<locals>" not in qualified
        and all(part.isidentifier() for part in qualified.split("."))
    ):
        # Keep each transcript line independently replayable.  Common command
        # support types are available as ``commands.X``; ecosystem-owned
        # material/geometry records use an explicit lazy import instead of
        # relying on an invisible console import history.
        root, *members = qualified.split(".")
        expression = (
            f"__import__({module!r}, fromlist=[{root!r}]).{qualified}"
        )
        return expression, True
    return kind.__name__, False


def _value_expression(value: Any, *, depth: int = 0) -> tuple[str, bool]:
    if depth > _MAX_DEPTH:
        return "...", False
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return repr(value), True
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value), True
        name = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return f"float({name!r})", True
    if isinstance(value, np.generic):
        return _value_expression(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        payload, exact = _value_expression(value.tolist(), depth=depth + 1)
        return f"np.array({payload})", exact
    if isinstance(value, Enum):
        name, available = _qualified_type(value)
        return f"{name}.{value.name}", available
    if isinstance(value, dict):
        if len(value) > _MAX_ITEMS:
            return f"{{...}}  # {len(value)} entries", False
        parts = []
        exact = True
        for key, item in value.items():
            key_text, key_exact = _value_expression(key, depth=depth + 1)
            item_text, item_exact = _value_expression(item, depth=depth + 1)
            parts.append(f"{key_text}: {item_text}")
            exact = exact and key_exact and item_exact
        return "{" + ", ".join(parts) + "}", exact
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) > _MAX_ITEMS:
            return f"[...]  # {len(value)} items", False
        rendered = [_value_expression(item, depth=depth + 1) for item in value]
        exact = all(item[1] for item in rendered)
        body = ", ".join(item[0] for item in rendered)
        if isinstance(value, tuple):
            return "(" + body + ("," if len(value) == 1 else "") + ")", exact
        if isinstance(value, list):
            return "[" + body + "]", exact
        constructor = "set" if isinstance(value, set) else "frozenset"
        return f"{constructor}([{body}])", exact
    if is_dataclass(value):
        name, exact = _qualified_type(value)
        arguments = []
        for item in fields(value):
            if not item.init:
                continue
            current = getattr(value, item.name)
            has_default, default = _default_value(item)
            if has_default:
                try:
                    if current == default:
                        continue
                except (TypeError, ValueError):
                    pass
            rendered, available = _value_expression(current, depth=depth + 1)
            arguments.append(f"{item.name}={rendered}")
            exact = exact and available
        return f"{name}({', '.join(arguments)})", exact
    representation = repr(value)
    if len(representation) > 160:
        representation = representation[:157] + "..."
    return representation, False


def _command_expression(command: Command) -> tuple[str, bool]:
    if isinstance(command, CompositeCommand):
        # ``run_many`` accepts unevaluated Command instances.  Calling
        # ``commands.run`` for each child here would execute it immediately
        # and pass its return value (for example a PlateSection) into the
        # outer batch, where the stack would then try ``value.do(project)``.
        rendered = [_value_expression(item) for item in command.commands]
        body = ",\n    ".join(item[0] for item in rendered)
        return (
            "commands.run_many([\n    " + body + f"\n], label={command.label!r})",
            all(item[1] for item in rendered),
        )
    expression, exact = _value_expression(command)
    return f"commands.run({expression})", exact


def command_to_python(command: Command) -> str:
    """Return one replay-oriented Python line for a successful command."""

    expression, exact = _command_expression(command)
    if len(expression) > _MAX_LINE:
        return f"# {command.label}: command omitted ({len(expression)} characters)"
    if exact:
        return expression
    return f"# Review before replay: {expression}"


def command_event_to_python(event: CommandEvent) -> str:
    """Format run/undo/redo exactly as exposed by the scripting context."""

    if event.action == "undo":
        return f"commands.undo()  # {event.command.label}"
    if event.action == "redo":
        return f"commands.redo()  # {event.command.label}"
    return command_to_python(event.command)

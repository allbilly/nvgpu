"""Falcon state oracle — instruction-accurate, event-driven executor."""

__version__ = "0.1.0"

from .errors import FalconOracleError
from .state import FalconState
from .trace import EventKind, OracleEvent

__all__ = [
    "FalconOracleError",
    "FalconState",
    "EventKind",
    "OracleEvent",
    "__version__",
]

"""Shared types for context-lens classifiers."""
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class ClassifierResult:
    """Output of a single classifier's detect() call."""

    pattern_name: str
    detected: bool
    severity: Severity
    confidence: float   # 0.0–1.0
    evidence: dict


@dataclass
class Recommendation:
    """Actionable fix for a detected pattern."""

    pattern_name: str
    description: str
    code_before: str    # what not to do
    code_after: str     # the fix
    estimated_recovery: str

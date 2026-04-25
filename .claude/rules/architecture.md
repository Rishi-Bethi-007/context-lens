# Architecture & Data Structures

## 6 Failure Patterns (implement in this order)
1. **lost_in_middle** — U-shaped curve, positions 30-70% ignored
2. **cliff_detector** — accuracy drops >20% between adjacent token counts
3. **tool_burial** — accuracy drops after 3rd+ sequential tool call
4. **instruction_drift** — early system prompt constraints weaken over time
5. **recency_bias** — model ignores everything except last N messages
6. **distractor** — similar-but-wrong info causes errors at long context

## Core Dataclasses

```python
@dataclass
class ProbeResult:
    position: float      # 0.0–1.0 where probe was planted
    token_count: int     # total tokens in context
    correct: bool
    response: str
    expected: str

@dataclass
class ClassifierResult:
    pattern_name: str
    detected: bool
    severity: Severity   # LOW | MEDIUM | HIGH
    confidence: float    # 0.0–1.0
    evidence: dict

@dataclass
class Recommendation:
    pattern_name: str
    description: str
    code_before: str     # what not to do
    code_after: str      # the fix
    estimated_recovery: str

@dataclass
class DiagnosticReport:
    agent_name: str
    timestamp: str
    measurement: MeasurementResult
    patterns: list[ClassifierResult]
    recommendations: list[Recommendation]
    overall_score: str   # A/B/C/D/F
    degradation_cliff_tokens: int | None

    def show(self) -> None: ...   # open in browser
    def save(self, path: str) -> None: ...
    def summary(self) -> None: ...  # print to terminal
```

## Probe Injection Rules
- Probe facts must NOT already exist in context (no false positives)
- Never inject mid-sentence or mid-tool-result — always at message boundaries
- Use tiktoken for token counting, not word count
- Run each probe 3x and majority-vote to reduce noise

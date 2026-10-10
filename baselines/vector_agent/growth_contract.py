"""Distinguish an explicitly stated report comparative from a fixed table base.

The recognizer covers one literal contract clause, with case/whitespace normalization.
It does not infer denominator vintage from a target name or parse arbitrary formulas.
"""

_REPORT_CLAUSE = "using both figures exactly as printed in that report"
_TABLE_START = "3. Values in the entity table are authoritative."
_LEVEL_START = "Forecast the level from the most recent levels"
_REPORT_RULE = (
    "3. Follow the task's comparison vintage. When both figures come from the resolving "
    "report, forecast both the target-period level and that report's comparative using only "
    "pre-cutoff evidence, then compute the task-defined growth. The given table comparative "
    "is a pre-cutoff anchor; it may differ from the future printed comparative after "
    "restatement. Do not treat the future comparative as already observed or automatically "
    "replace it with the table value. "
)


def report_comparative_target(task: dict) -> bool:
    prompt = " ".join(str(task.get("prompt", "")).casefold().split())
    return _REPORT_CLAUSE in prompt


def forecast_method(method: str, task: dict) -> str:
    """Replace only fixed-base instructions for the recognized report contract."""
    if not report_comparative_target(task):
        return method
    before, rest = method.split(_TABLE_START, 1)
    _, after = rest.split(_LEVEL_START, 1)
    before = before.replace("(the level stays at the table's base value)",
                            "(the level equals its forecasted resolving-report comparative)")
    return before + _REPORT_RULE + _LEVEL_START + after

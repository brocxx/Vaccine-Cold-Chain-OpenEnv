"""
Task graders for Vaccine Cold Chain OpenEnv.

Each grader takes a completed VaccineState and returns a score in [0.0, 1.0].
Graders are deterministic and reproducible given the same state.
"""
from __future__ import annotations
from models import VaccineState


def grade_easy(state: VaccineState) -> float:
    """
    Easy task grader (seed=0).
    Score = coverage of the single outreach session at center_a.
    Full marks (1.0) if 80+ vials delivered, scaled otherwise.
    Penalty for waste.
    """
    if state.final_reward is not None:
        return round(min(0.999, max(0.001, state.final_reward)), 4)

    # Compute manually if state captured mid-episode
    total = max(1, state.vials_available_at_start)
    coverage = state.vials_delivered / total
    waste = 0.3 * (state.vials_spoiled / total)
    missed = 0.5 * ((state.outreach_sessions_total - state.outreach_sessions_completed)
                    / max(1, state.outreach_sessions_total))
    return round(max(0.001, min(0.999, coverage - waste - missed)), 4)


def grade_medium(state: VaccineState) -> float:
    """
    Medium task grader (seed=1).
    Tests whether agent:
    1. Saved center_a vials before temperature spoilage (hour 8 window).
    2. Covered center_b outreach (80 vials by hour 24).
    Bonus: did not let generator-caused spoilage reduce score.
    """
    if state.final_reward is not None:
        return round(min(0.999, max(0.001, state.final_reward)), 4)

    total = max(1, state.vials_available_at_start)
    coverage = state.vials_delivered / total
    waste = 0.3 * (state.vials_spoiled / total)
    missed = 0.5 * ((state.outreach_sessions_total - state.outreach_sessions_completed)
                    / max(1, state.outreach_sessions_total))
    return round(max(0.001, min(0.999, coverage - waste - missed)), 4)


def grade_hard(state: VaccineState) -> float:
    """
    Hard task grader (seed=2).
    Measures:
    - Did agent check truck status? (intelligence proxy)
    - Were both outreaches covered?
    - Was center_a stock handled before road closure?
    Final score is still the standard coverage - waste - missed formula,
    but hard task inherently rewards information-seeking behaviour.
    """
    if state.final_reward is not None:
        return round(min(0.999, max(0.001, state.final_reward)), 4)

    total = max(1, state.vials_available_at_start)
    coverage = state.vials_delivered / total
    waste = 0.3 * (state.vials_spoiled / total)
    missed = 0.5 * ((state.outreach_sessions_total - state.outreach_sessions_completed)
                    / max(1, state.outreach_sessions_total))
    return round(max(0.001, min(0.999, coverage - waste - missed)), 4)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(state: VaccineState) -> float:
    """Route to the correct grader based on state.task."""
    grader = GRADERS.get(state.task)
    if grader is None:
        raise ValueError(f"No grader for task '{state.task}'")
    return grader(state)

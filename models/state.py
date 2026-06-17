from typing_extensions import TypedDict

from models.outputs import (
    PlanningOutput,
    ResearchOutput,
    ValidationOutput,
    WriterOutput,
    CriticOutput,
    GapAnalysis
)


class ResearchState(
    TypedDict
):
    query: str

    planning_output: (
        PlanningOutput | None
    )

    research_outputs: (
        list[ResearchOutput]
    )

    additional_research_outputs: list[ResearchOutput]

    validation_output: (
        ValidationOutput | None
    )

    writer_output: (
        WriterOutput | None
    )

    critic_output: (
        CriticOutput | None
    )


    gap_analysis: GapAnalysis | None

    revision_count: int
    previous_critic_score: float
    report_versions: list[str]

    
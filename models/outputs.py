from pydantic import BaseModel, Field


class PlanningOutput(BaseModel):
    topic: str = Field(
        ...,
        description="Research topic"
    )

    objective: str = Field(
        ...,
        description="Main research objective"
    )

    research_questions: list[str] = Field(
        default_factory=list,
        description="Questions to answer"
    )

    sections: list[str] = Field(
        default_factory=list,
        description="Sections for final report"
    )

    search_queries: list[str] = Field(
        default_factory=list,
        description="Search queries for researcher agent"
    )


from pydantic import BaseModel, Field


class ResearchOutput(BaseModel):
    search_query: str = Field(
        ...,
        description="Search query used for research"
    )

    summary: str = Field(
        ...,
        description="Research findings summary"
    )

    key_findings: list[str] = Field(
        default_factory=list,
        description="Important findings"
    )

    sources: list[str] = Field(
        default_factory=list,
        description="Source URLs"
    )


class ValidationOutput(BaseModel):
    is_valid: bool

    overall_score: float

    coverage_score: float

    relevance_score: float

    depth_score: float

    source_quality_score: float

    feedback: str

    improvement_suggestions: list[str]


class WriterOutput(BaseModel):
    report: str


class CriticOutput(BaseModel):
    critique: str

    final_report: str
from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import ValidationOutput

from models.state import ResearchState

VALIDATION_PROMPT = """
You are an expert research quality evaluator.

Evaluate the research based on:

1. Coverage
2. Relevance
3. Research Depth
4. Source Quality

Scoring Rules:

0-10 scale.

Return:

- overall_score
- coverage_score
- relevance_score
- depth_score
- source_quality_score
- feedback
- improvement_suggestions
- is_valid

Set is_valid=True only if overall_score >= 8.
"""

def validator_agent(
    state: ResearchState
) -> ResearchState:

    planning_output = state[
        "planning_output"
    ]

    research_outputs = state[
        "research_outputs"
    ]

    structured_llm = (
        llm.with_structured_output(
            ValidationOutput
        )
    )

    research_text = ""

    for output in research_outputs:

        research_text += f"""

SEARCH QUERY:
{output.search_query}

SUMMARY:
{output.summary}

KEY FINDINGS:
{output.key_findings}

SOURCES:
{output.sources}

"""

    validation_result = (
        structured_llm.invoke(
            [
                HumanMessage(
                    content=f"""
{VALIDATION_PROMPT}

TOPIC:
{planning_output.topic}

OBJECTIVE:
{planning_output.objective}

RESEARCH QUESTIONS:
{planning_output.research_questions}

SECTIONS:
{planning_output.sections}

RESEARCH DATA:
{research_text}
"""
                )
            ]
        )
    )

    state[
        "validation_output"
    ] = validation_result

    return state
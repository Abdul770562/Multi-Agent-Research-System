from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import (
    CriticOutput
)

from models.state import (
    ResearchState
)

CRITIC_PROMPT = """
You are a senior research reviewer.

Evaluate the report on:

1. Structure
2. Clarity
3. Research Quality
4. Completeness

Provide:

- scores from 0 to 10
- strengths
- weaknesses
- improvement suggestions

Rules:

- Be critical and objective.
- Identify missing information.
- Identify weak arguments.
- Identify repetitive content.
- Identify areas lacking evidence.

"""

def critic_agent(
    state: ResearchState
) -> ResearchState:

    planning_output = state[
        "planning_output"
    ]

    validation_output = state[
        "validation_output"
    ]

    writer_output = state[
        "writer_output"
    ]

    structured_llm = (
        llm.with_structured_output(
            CriticOutput
        )
    )

    critic_result = (
        structured_llm.invoke(
            [
                HumanMessage(
                    content=f"""
{CRITIC_PROMPT}

TOPIC:
{planning_output.topic}

OBJECTIVE:
{planning_output.objective}

VALIDATION FEEDBACK:
{validation_output.feedback}

REPORT TITLE:
{writer_output.title}

EXECUTIVE SUMMARY:
{writer_output.executive_summary}

REPORT:
{writer_output.report}
"""
                )
            ]
        )
    )

    critic_result.approved = (
    critic_result.overall_score >= 8
    )

    old_critic = state.get(
        "critic_output"
    )

    if old_critic:

        state[
            "previous_critic_score"
        ] = (
            old_critic.overall_score
        )

    state[
        "critic_output"
    ] = critic_result

    print(
    f"Critic Score: "
    f"{critic_result.overall_score}"
    )

    return state
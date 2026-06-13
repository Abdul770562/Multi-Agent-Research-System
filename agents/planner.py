from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from config.llm import llm
from models.outputs import PlanningOutput
from models.state import ResearchState


PLANNER_SYSTEM_PROMPT = """
You are an expert research planning agent.

Your responsibility is to create a structured research plan.

Given a research topic, generate:

1. topic
2. objective
3. research_questions
4. sections 
5. search_queries

Rules:

- Create 5 to 8 high-quality research questions.
- Create 5 to 8 report sections.
- Create 8 to 12 search queries.
- Search queries should be highly specific.
- Sections should cover the topic comprehensively.
- Objective should be concise and actionable.

Return ONLY valid JSON.

Do not include markdown.

Do not include explanations.
"""


def planner_agent(state: ResearchState) -> ResearchState:
    """
    Planner Agent

    Generates a structured research plan
    from the user's topic.
    """

    topic = state["query"]

    structured_llm = llm.with_structured_output(
        PlanningOutput,
        method="json_mode"
    )

    try:
        planning_output = structured_llm.invoke(
            [
                HumanMessage(
                    content=f"""
{PLANNER_SYSTEM_PROMPT}

Research Topic:
{topic}
"""
                )
            ]
        )

        state["planning_output"] = planning_output
        print("Planner completed")

        return state

    except ValidationError as e:
        raise RuntimeError(
            f"Planner output validation failed: {e}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Planner agent failed: {e}"
        )
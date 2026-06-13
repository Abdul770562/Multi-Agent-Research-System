from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import (
    WriterOutput
)

from models.state import (
    ResearchState
)

WRITER_PROMPT = """
You are a senior research analyst.

Generate a detailed research report.

Requirements:

- Minimum 1200 words.
- Use markdown.
- Include:
    Executive Summary
    Introduction
    Main Sections
    Conclusion
    References
- Use all provided key findings.
- Do not repeat information.
- Do not invent facts.
- Cite evidence from the research findings.
- Use professional academic language.

"""

def writer_agent(
    state: ResearchState
) -> ResearchState:

    planning_output = state[
        "planning_output"
    ]

    research_outputs = state[
        "research_outputs"
    ]

    validation_output = state[
        "validation_output"
    ]

    if not validation_output.is_valid:

        print(
            "WARNING: Research failed validation. "
            "Generating report anyway."
        )

    structured_llm = (
        llm.with_structured_output(
            WriterOutput
        )
    )

    research_context = ""

    all_sources = set()

    for research in research_outputs:

        research_context += f"""

SEARCH QUERY:
{research.search_query}

SUMMARY:
{research.summary}

KEY FINDINGS:
{research.key_findings}

"""

        for source in research.sources:
            all_sources.add(source)

    writer_output = (
        structured_llm.invoke(
            [
                HumanMessage(
                    content=f"""
{WRITER_PROMPT}

TOPIC:
{planning_output.topic}

OBJECTIVE:
{planning_output.objective}

REPORT SECTIONS:
{planning_output.sections}

VALIDATION FEEDBACK:
{validation_output.feedback}

RESEARCH:
{research_context}

REFERENCES:
{list(all_sources)}
"""
                )
            ]
        )
    )

    references_section = "\n\n## References\n\n"

    for idx, source in enumerate(
        sorted(all_sources),
        start=1
    ):
        references_section += (
            f"{idx}. {source}\n"
        )

    writer_output.report += references_section

    writer_output.references = (
        list(all_sources)
    )

    state[
        "writer_output"
    ] = writer_output

    state["report_versions"].append(
    writer_output.report
    )

    print("Writer completed")

    return state
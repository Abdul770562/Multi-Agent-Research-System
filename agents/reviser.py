from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import WriterOutput
from models.state import ResearchState
import re

REVISION_PROMPT = """
You are a senior research editor.

Your task is to improve an existing report.

CRITICAL GROUNDING RULES:

You may ONLY use information contained in:

1. Current Report
2. Research Findings

You MUST NOT:

- Invent sources
- Invent citations
- Invent studies
- Invent statistics
- Invent percentages
- Invent case studies
- Invent organizations
- Invent facts

If information does not exist in the
research findings, do not add it.

EDITOR RULES:

- Preserve report structure.
- Preserve valid content.
- Improve clarity.
- Improve completeness.
- Improve depth where evidence exists.
- Address every weakness.
- Address every improvement suggestion.
- Expand sections using ONLY provided evidence.

Return an improved report.

Do not create new references.
Do not create new evidence.

You must not:

- add references
- remove references
- rename references
- create a new references section

For every weakness:

1. Identify the section that should be improved.
2. Modify that section.
3. Explain the improvement internally.
4. Preserve all other sections.
"""

def reviser_agent(
    state: ResearchState
) -> ResearchState:

    writer_output = state[
        "writer_output"
    ]

    critic_output = state[
        "critic_output"
    ]

    research_outputs = state[
        "research_outputs"
    ]

    research_context = ""

    for item in research_outputs:

        findings = "\n".join(
            f"- {finding}"
            for finding in item.key_findings
        )

        sources = "\n".join(
            f"- {source}"
            for source in item.sources
        )

        research_context += f"""

SEARCH QUERY:
{item.search_query}

SUMMARY:
{item.summary}

KEY FINDINGS:
{findings}

SOURCES:
{sources}

"""

    structured_llm = (
        llm.with_structured_output(
            WriterOutput
        )
    )

    allowed_sources = set()

    for item in research_outputs:
        allowed_sources.update(
            item.sources
        )

    revised_output = (
        structured_llm.invoke(
            [
                HumanMessage(
                    content=f"""
    {REVISION_PROMPT}

    CURRENT REPORT:

    {writer_output.report}

    RESEARCH FINDINGS:

    {research_context}

    ALLOWED SOURCES:

    {list(allowed_sources)}

    WEAKNESSES:

    {critic_output.weaknesses}

    IMPROVEMENT SUGGESTIONS:

    {critic_output.improvement_suggestions}
    """
                )
            ]
        )
    )

    urls_found = set(
        re.findall(
            r"https?://[^\s)\]]+",
            revised_output.report
        )
    )

    invalid_urls = (
        urls_found - allowed_sources
    )

    if invalid_urls:
        print(
        "WARNING: Reviser introduced "
        "unauthorized sources:"
        )

        print(invalid_urls) 


    revised_output.references = (
        writer_output.references
    )

    for url in invalid_urls:
        revised_output.report = (
            revised_output.report.replace(
                url,
                "[REMOVED_UNAUTHORIZED_SOURCE]"
            )
        )

    state[
        "writer_output"
    ] = revised_output

    state[
        "revision_count"
    ] += 1

    state[
        "report_versions"
    ].append(
        revised_output.report
    )

    
    

    print(
    f"Revision count: "
    f"{state['revision_count']}"
    )

    return state
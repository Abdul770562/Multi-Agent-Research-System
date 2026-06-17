from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import (
    ResearchOutput
)

from models.state import (
    ResearchState
)

from tools.search import (
    search_web
)

from tools.scraper import (
    scrape_url
)

RESEARCH_PROMPT = """
You are an expert research analyst.

Analyze the provided content.

Create:

1. concise summary
2. key findings

Focus on factual information.

Avoid speculation.
"""

def summarize_content(
    query: str,
    content: str
) -> ResearchOutput:

    structured_llm = llm.with_structured_output(
        ResearchOutput
    )

    return structured_llm.invoke(
        [
            HumanMessage(
                content=f"""
{RESEARCH_PROMPT}

SEARCH QUERY:
{query}

CONTENT:
{content}
"""
            )
        ]
    )



def research_query(
    query: str,
    max_urls: int = 2
) -> ResearchOutput | None:
    """
    Executes the complete research pipeline for a
    single search query.

    Returns:
        ResearchOutput if successful,
        otherwise None.
    """

    urls = search_web(
        query=query,
        max_results=max_urls
    )

    if not urls:
        return None

    combined_content = ""

    for url in urls:

        page_text = scrape_url(
            url
        )

        if not page_text:
            continue

        combined_content += (
            page_text + "\n\n"
        )

    if not combined_content:
        return None

    try:

        research_result = (
            summarize_content(
                query=query,
                content=combined_content
            )
        )

        research_result.sources = urls

        return research_result

    except Exception as e:

        print(
            f"Research failed for "
            f"{query}: {e}"
        )

        return None

def researcher_agent(
    state: ResearchState
) -> ResearchState:

    planning_output = state[
        "planning_output"
    ]

    research_outputs = []

    MAX_SEARCH_QUERIES = 3

    for query in planning_output.search_queries[:MAX_SEARCH_QUERIES]:

        print(
            f"\nResearching: {query}"
        )

        result = research_query(
            query=query
        )

        if result:

            research_outputs.append(
                result
            )

    state[
        "research_outputs"
    ] = research_outputs

    print("Researcher completed")

    return state

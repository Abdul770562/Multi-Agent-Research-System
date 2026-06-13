from agents.planner import (
    planner_agent
)

from agents.researcher import (
    researcher_agent
)

from agents.validator import (
    validator_agent
)

from agents.writer import (
    writer_agent
)

state = {
    "query":
    "Artificial Intelligence in Healthcare",

    "planning_output":
    None,

    "research_outputs":
    [],

    "validation_output":
    None,

    "writer_output":
    None,

    "critic_output":
    None
}

state = planner_agent(state)

state = researcher_agent(state)

state = validator_agent(state)

state = writer_agent(state)

print(
    state["writer_output"]
)
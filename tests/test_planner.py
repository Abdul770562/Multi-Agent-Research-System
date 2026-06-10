from agents.planner import planner_agent


state = {
    "query": "Artificial Intelligence in Healthcare",

    "planning_output": None,

    "research_outputs": [],

    "validation_output": None,

    "writer_output": None,

    "critic_output": None
}

result = planner_agent(state)

print(result["planning_output"])
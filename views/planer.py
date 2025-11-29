from . import ollama as ollama_views
from models.Answer import *
from models.ollama import OllamaOptions
from pydantic import BaseModel
from .llm_planer import GLOBAL_ROUTING_SPEC


class MarkdownPlan(BaseModel):
    route: int  # must be 0, 1, 2, or 3
    plan_markdown: str  # full markdown plan text


def llm_planner(query: str, route: int, model: str = "llama3:latest") -> JSONFormat:
    """
    Agent #4: planner.

    Called AFTER the routing pipeline has already decided the final route (0..3).
    This agent does NOT change the route. It produces a full Markdown plan
    describing how the system will handle the request within the chosen route.

    Returns JSON:
    {
        "route": <int>,              # 0, 1, 2, or 3 (must match the input route)
        "plan_markdown": "<markdown>"  # human-readable plan in Markdown format
    }
    """

    system_prompt = "\n".join(
        [
            "You are Agent #4: PLANNER.",
            "You receive two things:",
            "- the user message (single turn, no chat history),",
            "- the FINAL route code (0, 1, 2, or 3) selected by the routing pipeline.",
            "",
            "Your ONLY job is to produce a clear, human-readable Markdown PLAN that",
            "explains how the system will handle this request inside the chosen route.",
            "You MUST NOT change the route. You MUST NOT re-classify the request.",
            "",
            "GLOBAL ROUTING CONTRACT (must stay consistent with all previous agents):",
            GLOBAL_ROUTING_SPEC,
            "",
            "INPUT FORMAT YOU RECEIVE (as plain text):",
            "[ROUTE]",
            "<one integer in {0,1,2,3}>",
            "",
            "[USER_INPUT]",
            "<the normalized user message>",
            "",
            "INTERPRETATION OF ROUTE:",
            "- 0 = SHALLOW TEXT MODEL",
            "- 1 = DEEP WEB RESEARCH",
            "- 2 = DOCS PIPELINE",
            "- 3 = IMAGES PIPELINE",
            "",
            "WHAT YOUR MARKDOWN PLAN SHOULD LOOK LIKE:",
            "- It MUST be a well-structured Markdown document.",
            "- Use headings (e.g., '#', '##') and ordered or unordered lists.",
            "- The plan is user-facing: it explains to the user what steps the system",
            "  will follow to answer their request.",
            "- You can include sections like 'Goal', 'Approach', 'Steps' etc.",
            "- 3–7 steps is usually a good range.",
            "",
            "ROUTE-SPECIFIC GUIDELINES:",
            "",
            "If route = 0 (SHALLOW TEXT MODEL):",
            "- DO NOT mention web search, external APIs, documents, or images.",
            "- Focus on reasoning with internal model knowledge only.",
            "- Explain that the answer will be based on existing knowledge and logical",
            "  explanation.",
            "",
            "If route = 1 (DEEP WEB RESEARCH):",
            "- Clearly state that the system will use web search and online sources.",
            "- Emphasize finding recent / up-to-date information, cross-checking and",
            "  synthesizing multiple sources.",
            "",
            "If route = 2 (DOCS PIPELINE):",
            "- Clearly state that the system will rely on the user's documents (PDFs/text).",
            "- Explain that it will retrieve relevant parts of the documents and answer",
            "  based on them.",
            "",
            "If route = 3 (IMAGES PIPELINE):",
            "- Clearly state that the system will rely on the user's images/photos.",
            "- Explain that it will analyze the image content and answer based on what",
            "  is visible there.",
            "",
            "IMPORTANT RULES:",
            "- NEVER override or reinterpret the route. Use it as given.",
            "- Do NOT mention internal agent names or implementation details.",
            "- The Markdown text MUST be in English.",
            "",
            "OUTPUT FORMAT (STRICT):",
            "- You MUST return ONLY raw JSON matching this structure:",
            '  {"route": int, "plan_markdown": str}',
            "- route MUST be exactly the same integer you received in [ROUTE].",
            "- plan_markdown MUST contain the FULL Markdown document (no truncation).",
            "- Do NOT output plain Markdown alone; always wrap it in JSON.",
        ]
    )

    planner_input = "\n".join(
        [
            "[ROUTE]",
            str(route),
            "",
            "[USER_INPUT]",
            query or "",
        ]
    )

    format: JSONFormat = JSONFormat(
        answer=Answer(
            query=planner_input,
            other_dict=[
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ],
        ),
        format=MarkdownPlan,
    )

    return ollama_views.json_output(
        query=format,
        model=model,
        options=OllamaOptions(temperature=0),
    )


if __name__ == "__main__":
    while True:
        user_query = input("input>>> ").strip()
        if not user_query:
            break

        route_str = input("route (0-3)>>> ").strip()
        try:
            route = int(route_str)
        except ValueError:
            print("Route must be an integer 0, 1, 2, or 3.")
            continue

        if route not in (0, 1, 2, 3):
            print("Route must be one of: 0, 1, 2, 3.")
            continue

        result = llm_planner(user_query, route)
        # result.output.plan_markdown -> готовий Markdown-план
        print(result.output)

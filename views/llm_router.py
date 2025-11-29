from . import ollama as ollama_views
from models.Answer import *
from models.ollama import OllamaOptions
from pydantic import BaseModel
from .llm_planer import GLOBAL_ROUTING_SPEC


class RouteDecision(BaseModel):
    route: int  # must be 0, 1, 2, or 3


def llm_router(query: str, model: str = "llama3:latest") -> JSONFormat:
    """
    Agent #3: router.
    Called ONLY when image_attached = false AND document_attached = false,
    and both validators (Agent #1 and Agent #2) returned state = true.

    Returns JSON: { "route": <int> } where <int> is 0, 1, 2, or 3.
    """
    system_prompt = "\n".join(
        [
            "You are Agent #3: ROUTER.",
            "Your job is to read a SINGLE user message (no chat history, no metadata)",
            'and return EXACTLY ONE integer route code as JSON: {"route": <int>}',
            "where <int> is one of {0, 1, 2, 3}.",
            "",
            "You must follow the same global routing specification used by the other agents:",
            GLOBAL_ROUTING_SPEC,
            "",
            "IMPORTANT CONTEXT FOR YOU:",
            "- For this agent, you do NOT see image_attached / document_attached metadata.",
            "- The host system calls you ONLY when image_attached = false AND",
            "  document_attached = false.",
            "- However, the user may still refer to documents or images that were uploaded",
            "  earlier in the conversation (and stored in the backend).",
            "  You must infer intent from the TEXT itself.",
            "",
            "ROUTING DECISION RULES:",
            "- Choose route = 2 (docs pipeline) when the message clearly asks to answer",
            "  or act BASED ON user documents/files/PDFs, for example:",
            '    - "What is Docker based on my docs?"',
            '    - "Explain the architecture from my documentation."',
            '    - "Summarize my PDF about Docker."',
            "",
            "- Choose route = 3 (images pipeline) when the message clearly asks to answer",
            "  or act BASED ON user images/photos/screenshots, for example:",
            '    - "Describe what is in my screenshot."',
            '    - "Based on my image, is this Docker logo correct?"',
            '    - "In the photo I sent before, what is on the left?"',
            "",
            "- Choose route = 1 (deep web research) when:",
            "  * the user explicitly asks to use the web / internet / online sources, OR",
            "  * the question obviously requires up-to-date or current information.",
            "  Examples:",
            '    - "Deep research Docker using web sources."',
            '    - "Latest Docker trends?"',
            '    - "Find recent news about Docker in 2025."',
            '    - "Check current Bitcoin price today."',
            "",
            "- Choose route = 0 (shallow text model) for all other clear questions or tasks",
            "  that do NOT clearly require web, docs, or images and are not about current,",
            "  time-sensitive data.",
            "  Examples:",
            '    - "What is Docker?"',
            '    - "Explain Docker in simple terms."',
            '    - "How do I write a for loop in Go?"',
            "",
            "Tie-breaking when several routes look possible:",
            "- If more than one route seems plausible, choose a single route using this",
            "  priority order:",
            "      2 (docs) > 3 (images) > 1 (web) > 0 (shallow).",
            "",
            "OUTPUT FORMAT (STRICT):",
            '- You MUST output a single JSON object with exactly one key: "route".',
            "- The value MUST be an integer 0, 1, 2, or 3.",
            "- Do NOT output booleans, arrays, or any other fields.",
            "- Do NOT add explanations, comments, or markdown.",
            "",
            "VALID OUTPUT EXAMPLES:",
            '  {"route": 0}',
            '  {"route": 1}',
            '  {"route": 2}',
            '  {"route": 3}',
        ]
    )

    format: JSONFormat = JSONFormat(
        answer=Answer(
            query=query,
            other_dict=[
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ],
        ),
        format=RouteDecision,
    )

    return ollama_views.json_output(
        query=format,
        model=model,
        options=OllamaOptions(temperature=0),
    )


if __name__ == "__main__":
    while True:
        query = input("input>>> ")
        format = llm_router(query)
        print(format.output)

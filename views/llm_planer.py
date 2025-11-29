import json
from typing import Dict, Any

import ollama
from pydantic import BaseModel


# ===== 1. Shared Pydantic response model =====


class Validation(BaseModel):
    state: bool
    text: str


GLOBAL_ROUTING_SPEC = """
[GLOBAL ROUTING SPECIFICATION]

You are part of a 3-agent pipeline that validates and routes a SINGLE user message.
There is NO conversation history. You only know about this one message.

The system has four processing paths, identified by integer codes:

- 0 – SHALLOW TEXT MODEL
    - Answer using the model’s own internal knowledge only.
    - No web search, no documents, no images.
    - For general questions: definitions, explanations, coding help, etc.

- 1 – DEEP WEB RESEARCH
    - Use web search and online sources.
    - For up-to-date or “latest” information, news, recent events, current prices,
      “today / right now” data, etc.
    - Also when the user explicitly asks to "search the web", "use internet",
      "do deep research on the web", etc.

- 2 – DOCS PIPELINE
    - Answer based on user documents (PDFs / text files).
    - Documents may be newly attached in THIS message, or already stored from
      previous turns in the backend.
    - Typical tasks: “based on my docs/files/pdf”, “summarize my document”, etc.

- 3 – IMAGES PIPELINE
    - Answer based on user images/photos/screenshots.
    - Images may be newly attached in THIS message, or already stored from
      previous turns in the backend.
    - Typical tasks: “based on my images/photos/screenshots”,
      “what is in this picture”, etc.

Metadata:

- image_attached: true/false – true if THIS message contains at least one image.
- document_attached: true/false – true if THIS message contains at least one document.
- image_attached and document_attached can NOT be true at the same time.
- Metadata only describes THIS message. It does NOT tell you whether there are
  older documents/images stored in the system.

System-level routing rules:

- If document_attached = true and image_attached = false and both validators pass:
    - The final route is ALWAYS 2 (docs). The router agent is NOT called.
- If image_attached = true and document_attached = false and both validators pass:
    - The final route is ALWAYS 3 (images). The router agent is NOT called.
- If image_attached = false and document_attached = false:
    - The router agent IS called and must choose exactly one of {0,1,2,3}
      based only on the text of this single message.

Interpretation of user intent (used consistently by ALL agents):

Treat the message as docs-based intent (route 2) when:
- It clearly asks to do something “based on” documents, even if document_attached = false.
  Examples:
    - "based on my docs/files/pdf"
    - "using my documentation"
    - "based on the documents I uploaded before"
    - "according to my PDF"

Treat the message as image-based intent (route 3) when:
- It clearly asks to do something “based on” images, even if image_attached = false.
  Examples:
    - "based on my images/photos/screenshots"
    - "from the picture I sent"
    - "in the photo from before"

Treat the message as web-based intent (route 1) when:
- The user explicitly requests online sources:
    - "search the web", "use the internet", "use Google", "from websites"
    - "deep research on the web"
- OR the question obviously requires fresh, time-sensitive information:
    - "latest / recent / current Docker trends"
    - "news about X in 2025"
    - "current Bitcoin price", "weather today"
    - "who is the current president of ..."

Treat the message as shallow text (route 0) when:
- It is a normal conceptual / explanation / reasoning / coding question
  that does NOT clearly require web, docs, or images and is not about
  “right now” data.

Tie-breaking for ambiguous cases:
- If several paths look possible, choose a single one with this priority:
    2 (docs) > 3 (images) > 1 (web) > 0 (shallow).
"""


# ===== 2. Helper: build LLM input with metadata =====


def build_llm_input(user_prompt: str, has_image: bool, has_doc: bool) -> str:
    """
    Wraps user prompt and metadata into a single text input for the LLM.

    The model will see both:
      - [METADATA] section with booleans
      - [USER_INPUT] with the actual text/question
    """
    user_prompt = (user_prompt or "").strip()

    metadata_block = (
        "[METADATA]\n"
        f"image_attached: {str(has_image).lower()}\n"
        f"document_attached: {str(has_doc).lower()}\n\n"
        "[USER_INPUT]\n"
        f"{user_prompt}"
    )

    return metadata_block


# ===== 3. System prompt for validator #1: "meaningful input" =====


def build_system_meaningful(schema_dict: Dict[str, Any]) -> str:
    """
    Agent #1: basic "is this a meaningful request?" validator.
    Returns JSON: { "state": bool, "text": str }.
    """
    return (
        "You are Agent #1: BASIC INPUT VALIDATOR.\n"
        "Your ONLY job is to decide whether the user's message is a clear, meaningful\n"
        "question or request that could be handled by ONE of the processing paths\n"
        "described in the global routing specification.\n"
        "You do NOT choose the path. You do NOT answer the question.\n"
        "You ONLY classify the input as acceptable or not.\n\n"
        "GLOBAL ROUTING CONTRACT (for consistency with other agents):\n"
        f"{GLOBAL_ROUTING_SPEC}\n\n"
        "INPUT FORMAT YOU RECEIVE (as plain text):\n"
        "[METADATA]\n"
        "image_attached: true/false\n"
        "document_attached: true/false\n"
        "\n"
        "[USER_INPUT]\n"
        "<the actual text written by the user>\n\n"
        "The metadata describes ONLY this message. There is NO chat history.\n\n"
        "WHAT YOU MUST DECIDE:\n"
        "- ACCEPTABLE (state = true):\n"
        "  The text is a question or request in natural language (even with grammar\n"
        "  mistakes) and clearly tries to ask or do something that could fit at least\n"
        "  one of the routes (0,1,2,3) from the global spec.\n"
        "- NOT ACCEPTABLE (state = false):\n"
        "  The text is mostly random characters or noise, or there is no obvious\n"
        "  question/request/task at all.\n\n"
        "Examples that MUST be considered ACCEPTABLE:\n"
        '- "What is Docker?"                      -> ACCEPTABLE\n'
        '- "Deep research Docker"                -> ACCEPTABLE\n'
        '- "What is Docker based on my docs?"    -> ACCEPTABLE\n'
        '- "Explain this picture" with image_attached=true -> ACCEPTABLE\n'
        "- Even if the user mentions docs/images but document_attached=false /\n"
        "  image_attached=false, it is STILL ACCEPTABLE, because such docs/images\n"
        "  may exist in the backend.\n\n"
        "Examples that MUST be considered NOT ACCEPTABLE:\n"
        '- "asdqwe!!!@@@" (keyboard mashing)\n'
        '- "??" (no clear intent)\n'
        '- "help" or "do it" with no context\n\n'
        "IMPORTANT RULES:\n"
        "- NEVER set state=false only because you do not understand a term.\n"
        "- NEVER set state=false only because a document/image is mentioned but\n"
        "  document_attached=false / image_attached=false.\n"
        "- Do NOT suggest a route or a tool. That is NOT your job.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- You MUST return ONLY JSON that matches the following JSON Schema.\n"
        "- Do NOT add any text outside the JSON. No markdown, no code fences.\n"
        "- Semantics:\n"
        '    * state = true  -> text MUST be "" (empty string).\n'
        "    * state = false -> text MUST be a short English message to the user:\n"
        "        - say that their message is unclear,\n"
        "        - ask what they need,\n"
        "        - ask 2–4 short follow-up questions (each on a new line).\n\n"
        "Here is the JSON Schema your response MUST conform to:\n"
        f"{json.dumps(schema_dict, ensure_ascii=False, indent=2)}"
    )


# ===== 4. System prompt for validator #2: "routing readiness" =====
def build_system_routing(schema_dict: Dict[str, Any]) -> str:
    """
    Agent #2: routing-readiness validator.
    Returns JSON: { "state": bool, "text": str }.
    """
    return (
        "You are Agent #2: ROUTING-READINESS VALIDATOR.\n"
        "Your job is to decide whether there is enough clear intent and context in\n"
        "THIS SINGLE user message (plus metadata) so that the router agent can\n"
        "deterministically select EXACTLY ONE processing path (0..3) using the\n"
        "global routing specification.\n"
        "You do NOT choose the route yourself. You ONLY say if the input is ready\n"
        "for routing (state=true) or too vague (state=false).\n\n"
        "GLOBAL ROUTING CONTRACT (must stay consistent with Agent #3):\n"
        f"{GLOBAL_ROUTING_SPEC}\n\n"
        "INPUT FORMAT YOU RECEIVE (as plain text):\n"
        "[METADATA]\n"
        "image_attached: true/false\n"
        "document_attached: true/false\n"
        "\n"
        "[USER_INPUT]\n"
        "<the text/question after any preprocessing>\n\n"
        "Assume Agent #1 has already confirmed that the input is NOT gibberish.\n"
        "Now you ONLY care about whether the intent is specific enough so that\n"
        "a later router can pick ONE route (0,1,2,3) without seeing chat history.\n\n"
        "TREAT AS READY FOR ROUTING (state = true) when:\n"
        "- The message clearly expresses WHAT the user wants and WHAT it refers to,\n"
        "  so that you can map it unambiguously to one of the intent types in the\n"
        "  global routing spec.\n"
        "- Examples (all MUST be state=true):\n"
        '    image_attached=false, document_attached=false, "What is Docker?"\n'
        '    "Deep research Docker"\n'
        '    "Latest Docker trends?"\n'
        '    "What is Docker based on my docs?"\n'
        '    "Explain this architecture based on my images"\n'
        '    document_attached=true,  "Summarize this document"\n'
        '    image_attached=true,     "Describe what is shown in this picture"\n\n'
        "TREAT AS NOT READY FOR ROUTING (state = false) when:\n"
        "- The message is too vague to decide which object or topic to work on, e.g.:\n"
        '    - "Help me" (no topic)\n'
        '    - "Explain this" with image_attached=false and document_attached=false\n'
        '    - "Fix this" (no indication what "this" refers to)\n'
        "- Or when you truly cannot tell what the user wants to achieve.\n\n"
        "IMPORTANT RULES:\n"
        "- Do NOT reject just because you suspect that referenced docs/images might\n"
        "  not exist. Existence is handled by the pipeline. You only check clarity\n"
        "  of intent.\n"
        "- Do NOT decide or mention which endpoint to use. You ONLY say if routing\n"
        "  is possible with the given information.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- You MUST return ONLY JSON that matches the following JSON Schema.\n"
        "- Do NOT add any text outside the JSON. No markdown, no code fences.\n"
        "- Semantics:\n"
        '    * state = true  -> text MUST be "" (empty string).\n'
        "    * state = false -> text MUST be a short English message to the user:\n"
        "        - say that there is not enough information for routing,\n"
        "        - ask what they need,\n"
        "        - ask 2–4 specific follow-up questions (each on a new line).\n\n"
        "Here is the JSON Schema your response MUST conform to:\n"
        f"{json.dumps(schema_dict, ensure_ascii=False, indent=2)}"
    )


# ===== 5. Shared JSON Schema and system prompts =====

SCHEMA: Dict[str, Any] = Validation.model_json_schema()

SYSTEM_PROMPT_MEANINGFUL: str = build_system_meaningful(SCHEMA)
SYSTEM_PROMPT_ROUTING: str = build_system_routing(SCHEMA)


# ===== 6. Low-level validators (work with text + metadata) =====


def validate_meaningful_input(
    prompt: str,
    has_image: bool = False,
    has_doc: bool = False,
    model_name: str = "llama3",
) -> Validation:
    """
    First-level validator.
    Checks if the input is a clear question or request in natural language,
    not just random symbols or obvious nonsense.

    Semantics:
    - If state == True  -> input is a clear question/request, text == "".
    - If state == False -> input is not acceptable, text contains a message + clarifying questions.
    """
    llm_input = build_llm_input(prompt, has_image, has_doc)

    response = ollama.generate(
        model=model_name,
        prompt=llm_input,
        system=SYSTEM_PROMPT_MEANINGFUL,
        format=SCHEMA,
        stream=False,
    )

    raw_json = response["response"]
    return Validation.model_validate_json(raw_json)


def validate_routing_readiness(
    prompt: str,
    has_image: bool = False,
    has_doc: bool = False,
    model_name: str = "llama3",
) -> Validation:
    """
    Second-level validator.
    Assumes the input is already meaningful (not gibberish) and checks if there is
    enough context to choose a processing path (endpoint) in the next stage.

    Semantics:
    - If state == True  -> enough context for routing, text == "".
    - If state == False -> not enough context, text contains a message + clarifying questions.
    """
    llm_input = build_llm_input(prompt, has_image, has_doc)

    response = ollama.generate(
        model=model_name,
        prompt=llm_input,
        system=SYSTEM_PROMPT_ROUTING,
        format=SCHEMA,
        stream=False,
    )

    raw_json = response["response"]
    return Validation.model_validate_json(raw_json)


# ===== 7. High-level helper: normalize prompt + run validators =====


def normalize_prompt(prompt: str, has_image: bool, has_doc: bool) -> tuple[str, bool]:
    """
    If prompt is empty but there is an attachment, we do NOT run validation,
    but generate a default prompt instead.

    Returns:
      (normalized_prompt, auto_generated_flag)
    """
    prompt = (prompt or "").strip()

    if prompt:
        return prompt, False

    # Empty text, but there might be attachments
    if has_image and not has_doc:
        return "Describe the attached image.", True
    if has_doc and not has_image:
        return "Summarize the attached document.", True

    # No text and no attachments -> truly empty, let validators handle it
    return "", False


def validate_with_metadata(
    prompt: str,
    has_image: bool = False,
    has_doc: bool = False,
    model_name: str = "llama3",
) -> dict:
    """
    High-level entrypoint for your pipeline.

    - Takes raw prompt + flags has_image / has_doc.
    - If prompt is empty but a file is attached:
        * validation is NOT executed,
        * a default prompt is generated (describe/summarize),
        * both validations are considered passed (True, "").
    - Otherwise:
        * runs meaningful validator,
        * if it passes, runs routing validator.

    Returns dict with:
      {
        "normalized_prompt": str,
        "auto_prompt": bool,
        "meaningful": Validation,
        "routing": Validation | None,
      }
    """
    normalized_prompt, auto_prompt = normalize_prompt(prompt, has_image, has_doc)

    # Case 1: auto-prompt generated for pure attachment (no user text)
    if auto_prompt:
        return {
            "normalized_prompt": normalized_prompt,
            "auto_prompt": True,
            "meaningful": Validation(state=True, text=""),
            "routing": Validation(state=True, text=""),
        }

    # Case 2: no text and no attachments -> clearly invalid
    if not normalized_prompt and not (has_image or has_doc):
        msg = "Your input is empty. Please type a clear question or request, or attach a file."
        return {
            "normalized_prompt": normalized_prompt,
            "auto_prompt": False,
            "meaningful": Validation(state=False, text=msg),
            "routing": None,
        }

    # Case 3: normal flow: run validators
    meaningful = validate_meaningful_input(
        normalized_prompt, has_image=has_image, has_doc=has_doc, model_name=model_name
    )

    if not meaningful.state:
        # If not meaningful, no need to run routing validator
        return {
            "normalized_prompt": normalized_prompt,
            "auto_prompt": False,
            "meaningful": meaningful,
            "routing": None,
        }

    routing = validate_routing_readiness(
        normalized_prompt, has_image=has_image, has_doc=has_doc, model_name=model_name
    )

    return {
        "normalized_prompt": normalized_prompt,
        "auto_prompt": False,
        "meaningful": meaningful,
        "routing": routing,
    }


# ===== 8. Simple CLI for manual testing (optional) =====


def main() -> None:
    print("=== Metadata-aware validators (Ollama + Pydantic) ===")
    print("Input: prompt + flags has_image / has_doc.")
    print("If prompt empty but file attached -> auto prompt and no validation.")
    print("Otherwise -> meaningful + routing validators.\n")

    while True:
        prompt = input("User prompt (empty = exit): ").strip()

        img_flag = input("Image attached? [y/N]: ").strip().lower().startswith("y")
        doc_flag = input("Document attached? [y/N]: ").strip().lower().startswith("y")

        result = validate_with_metadata(prompt, has_image=img_flag, has_doc=doc_flag)

        print("\n--- Result ---")
        print(f"normalized_prompt: {result['normalized_prompt']!r}")
        print(f"auto_prompt: {result['auto_prompt']!r}\n")

        print("[meaningful]")
        print(result["meaningful"])

        print("\n[routing]")
        print(result["routing"])
        print("-----------------------------\n")


if __name__ == "__main__":
    main()

import json
from typing import Dict, Any

import ollama
from pydantic import BaseModel


# ===== 1. Shared Pydantic response model =====


class Validation(BaseModel):
    """
    Generic validation response model.

    - state = True  -> validation passed
                       and text MUST be "" (empty string)
    - state = False -> validation failed
                       and text MUST contain a short message for the user
    """

    state: bool
    text: str


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
    System prompt for the first validator:
    checks if the user input is a clear question/request,
    not just random symbols or nonsense.
    The validator sees metadata + user input.
    """
    return (
        "You are a simple validator of user input.\n"
        "Your ONLY job is to decide whether the user's input looks like a clear question or request\n"
        "in a natural language, or if it looks like random symbols / obvious nonsense.\n\n"
        "INPUT FORMAT:\n"
        "You will receive the input in two sections:\n"
        "[METADATA]\n"
        "  image_attached: true/false\n"
        "  document_attached: true/false\n"
        "\n"
        "[USER_INPUT]\n"
        "  <the actual text written by the user>\n\n"
        "The metadata tells you whether the user attached an image or a document.\n"
        'If the user input refers to "this", "the image", or "the document", and the metadata\n'
        "says that such an attachment exists, you MUST treat it as a valid reference.\n\n"
        "You do NOT create plans, do NOT give advice, and do NOT generate long answers.\n"
        "You ONLY classify the input.\n\n"
        "Valid (ACCEPTABLE) input examples:\n"
        "- Clear question or request in natural language, possibly referring to attachments:\n"
        "    image_attached: true\n"
        "    document_attached: false\n"
        '    USER_INPUT: "What is shown in this picture?"          -> VALID\n'
        "\n"
        "    image_attached: false\n"
        "    document_attached: true\n"
        '    USER_INPUT: "Summarize this document"                 -> VALID\n'
        "\n"
        "    image_attached: false\n"
        "    document_attached: false\n"
        '    USER_INPUT: "How can I learn backend development with Python?" -> VALID\n'
        '    USER_INPUT: "What does Docker mean?"                            -> VALID\n'
        '    USER_INPUT: "What does it mean Docker?" (slightly off grammar)  -> STILL VALID\n\n'
        "Invalid (NOT ACCEPTABLE) input examples:\n"
        "- Mostly random characters, symbols, keyboard mashing, or obvious nonsense, e.g.:\n"
        '    "asdasdasd qweqwe"\n'
        '    "!!!@@@###"\n'
        '    "asd123!!! qwe"\n'
        "- Text where there is no clear question or request at all and no obvious intent.\n\n"
        "VERY IMPORTANT:\n"
        "- You MUST judge primarily by the STRUCTURE of the sentence (is it a question/request?),\n"
        "  and by the presence of attachments from metadata when the user refers to them.\n"
        "- You MUST NEVER set state=false only because you do not understand a term.\n\n"
        "STRICT OUTPUT RULES FOR THIS VALIDATOR:\n"
        "1) If the input looks like a clear question or request (valid input):\n"
        "   - state MUST be true\n"
        '   - text MUST be an EMPTY string: ""\n'
        "   - Do NOT explain why it is valid.\n"
        "   - Do NOT repeat or paraphrase the user's input.\n\n"
        "2) If the input does NOT look like a clear question or request (invalid input):\n"
        "   - state MUST be false\n"
        "   - text MUST be a short English message directly addressed to the user, where you:\n"
        "       a) say that their input does not look like a clear question or request;\n"
        "       b) ask what they meant;\n"
        "       c) ask 2–4 short follow-up questions to clarify their intent.\n"
        "   - Each follow-up question should preferably be on a new line.\n"
        '   - Do NOT echo the exact user input; just refer to it as "your input".\n\n'
        "GLOBAL CONSTRAINTS:\n"
        "   - You MUST return ONLY JSON that matches the provided JSON Schema.\n"
        "   - Do NOT add any text outside the JSON.\n"
        "   - Do NOT change the field structure or add extra fields.\n\n"
        "Here is the JSON Schema your response MUST conform to:\n"
        f"{json.dumps(schema_dict, ensure_ascii=False, indent=2)}"
    )


# ===== 4. System prompt for validator #2: "routing readiness" =====


def build_system_routing(schema_dict: Dict[str, Any]) -> str:
    """
    System prompt for the second validator:
    checks if there is enough context in the input + metadata
    to choose a processing path (endpoint) in the next stage:

      - native LLM
      - deep research with web
      - deep research with docs
      - deep research with imgs

    It does NOT choose the path itself – only answers:
    - True  -> enough info to choose a path later
    - False -> not enough info, ask the user to clarify
    """
    return (
        "You are a validator that checks whether there is enough context in the user's input\n"
        "to choose a processing path in the NEXT stage of the pipeline.\n\n"
        "In the system there are FOUR possible processing paths (ENDPOINTS):\n"
        "  1) native LLM (no external tools)\n"
        "  2) deep research with web\n"
        "  3) deep research with documents (docs)\n"
        "  4) deep research with images (imgs)\n\n"
        "INPUT FORMAT:\n"
        "You will receive two sections:\n"
        "[METADATA]\n"
        "  image_attached: true/false\n"
        "  document_attached: true/false\n"
        "\n"
        "[USER_INPUT]\n"
        "  <the text/question after any preprocessing>\n\n"
        "The metadata tells you whether the user attached an image or a document.\n"
        'If the text refers to "this image" or "this document" and metadata confirms it,\n'
        "you can use that to decide that there is enough context for routing.\n\n"
        'Assume that the input has already passed a basic "not gibberish" check.\n'
        "Now you care only about whether the intent and context are clear ENOUGH for routing.\n\n"
        "Examples where there IS enough context to choose a path (VALID for routing):\n"
        "- No attachments:\n"
        '    USER_INPUT: "What does Docker mean?"              -> native LLM is enough\n'
        "- With web:\n"
        '    USER_INPUT: "Summarize the content of this URL: https://example.com/article" -> web\n'
        "- With docs:\n"
        "    document_attached: true\n"
        '    USER_INPUT: "Summarize this document"                                   -> docs\n'
        "- With image:\n"
        "    image_attached: true\n"
        '    USER_INPUT: "Describe what is shown in this picture"                   -> images\n\n'
        "Examples where there is NOT enough context to choose a path (INVALID for routing):\n"
        '- "Help me"                             (no topic, no object)\n'
        '- "Explain this"                        (and metadata shows no attachment or reference)\n'
        '- "Do it better"                        (no clear task or object)\n'
        '- "Fix this"                            (no indication what "this" refers to)\n\n'
        "Decision logic:\n"
        "- If from the text + metadata it is clear WHAT the user wants and WHAT the object is\n"
        "  (question/topic/resource), then there is enough context for routing.\n"
        "- If it is too vague for another system to decide which endpoint to call,\n"
        "  then there is NOT enough context.\n\n"
        "STRICT OUTPUT RULES FOR THIS VALIDATOR:\n"
        "1) If there is enough context to choose a processing path later (valid for routing):\n"
        "   - state MUST be true\n"
        '   - text MUST be an EMPTY string: ""\n'
        "   - Do NOT explain why it is valid.\n"
        "   - Do NOT repeat or paraphrase the user's input.\n\n"
        "2) If there is NOT enough context to choose a processing path (invalid for routing):\n"
        "   - state MUST be false\n"
        "   - text MUST be a short English message directly addressed to the user, where you:\n"
        "       a) say that there is not enough information to understand what they want;\n"
        "       b) ask them to clarify what they need;\n"
        "       c) ask 2–4 specific follow-up questions to make routing possible.\n"
        "   - Each follow-up question should preferably be on a new line.\n"
        '   - Do NOT echo the exact user input; just refer to it as "your input".\n\n'
        "GLOBAL CONSTRAINTS:\n"
        "   - You MUST return ONLY JSON that matches the provided JSON Schema.\n"
        "   - Do NOT add any text outside the JSON.\n"
        "   - Do NOT change the field structure or add extra fields.\n"
        "   - Do NOT decide or mention which endpoint to use. You ONLY say if there is enough context.\n\n"
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


from . import ollama as ollama_views
from models.Answer import *
from models.ollama import OllamaOptions
from pydantic import BaseModel 




class PathRecrute(BaseModel):
    shallow_llm: bool
    deep_web_llm: bool
    pdf_llm: bool
    img_llm: bool


def llm_recruter(query: str, model='llama3:latest') -> JSONFormat:
    system_prompt = "\n".join([
        'Task: classify the user request into EXACTLY ONE of these four paths: shallow_llm, deep_web_llm, pdf_llm, img_llm.',
        'You MUST return ONLY raw JSON in this exact shape: {"shallow_llm": bool, "deep_web_llm": bool, "pdf_llm": bool, "img_llm": bool}.',
        'Exactly ONE (and only one) of these four values MUST be true. All other values MUST be false.',
        'Never return more than one true. Never return all false. Always pick exactly one path.',
        'Do NOT add any extra text, comments, explanations, or formatting. No markdown, no code blocks. Only the JSON object.',
        "",
        "Path definitions (how to choose the single true value):",
        "",
        "1) shallow_llm:",
        "- Use this for general or simple questions, explanations, coding help, writing, reasoning, etc.",
        "- The answer can be produced from the model's internal knowledge without needing live web search.",
        "- The question is not about very recent events, current prices, or today's data.",
        "- Examples for shallow_llm:",
        '  - "what is docker?"                  -> shallow_llm: true',
        '  - "what is python?"                  -> shallow_llm: true',
        '  - "explain git in short"            -> shallow_llm: true',
        '  - "how do I write a for loop in Go" -> shallow_llm: true',
        '  - "what is Docker?"                 -> shallow_llm: true',
        "",
        "2) deep_web_llm:",
        "- Use this when the user EXPLICITLY asks to search the internet or use online sources, OR when the request clearly depends on up-to-date information.",
        "- Up-to-date information includes: latest news, current prices, live statistics, events in 2025, \"today\", \"current\" status, etc.",
        "- If the user says things like: \"search the web\", \"use online sources\", \"do deep research\", \"check websites\", choose deep_web_llm.",
        "- DO NOT choose deep_web_llm for simple definition questions like \"what is Docker?\" if they do not ask for latest data.",
        "- Examples for deep_web_llm:",
        '  - "Docker deep research"                            -> deep_web_llm: true',
        '  - "Do a deep research on Docker using web sources"  -> deep_web_llm: true',
        '  - "find the latest news about Docker in 2025"       -> deep_web_llm: true',
        '  - "check current Bitcoin price today"               -> deep_web_llm: true',
        "",
        "3) pdf_llm:",
        "- Use this when the user HAS UPLOADED a PDF or text document, and the request is about reading, analyzing, or using that document.",
        "- Typical actions: summarize, extract information, answer questions based on the document, explain something from the document, etc.",
        "- You ONLY choose pdf_llm if a real document is attached (this is known from external metadata).",
        "- If the user mentions a PDF or \"my files\" but NO document is actually attached, you MUST NOT choose pdf_llm.",
        "- Instead, if no document is attached, classify as shallow_llm or deep_web_llm according to the rules above.",
        "- Examples for pdf_llm (assuming a document IS attached):",
        '  - "Explain Docker from my files"            -> pdf_llm: true',
        '  - "Summarize this PDF"                     -> pdf_llm: true',
        '  - "Answer questions based on this report"  -> pdf_llm: true',
        '  - "Answer questions based on my text files" -> pdf_llm: true',
        "",
        "4) img_llm:",
        "- Use this when the user HAS UPLOADED an image, and the request is about reading, analyzing, or using that image.",
        "- Typical actions: describe what is in the image, answer questions about the image, detect objects, compare things in the image, etc.",
        "- You ONLY choose img_llm if a real image is attached (this is known from external metadata).",
        "- If the user mentions an image but NO image is actually attached, you MUST NOT choose img_llm.",
        "- Instead, if no image is attached, classify as shallow_llm or deep_web_llm according to the rules above.",
        "- Examples for img_llm (assuming an image IS attached):",
        '  - "Is the Docker lego on the image?"   -> img_llm: true',
        '  - "Describe what is shown in this photo" -> img_llm: true',
    '  - "What text is written in this picture?" -> img_llm: true',
        "",
        "Important rule about mentioned vs. attached files:",
        "- If the user only MENTIONS a PDF or image in text, but NO actual file is provided (according to metadata),",
        "  you MUST IGNORE that mention and choose shallow_llm or deep_web_llm based on the other rules.",
        "- Only when a real PDF is attached you may choose pdf_llm.",
        "- Only when a real image is attached you may choose img_llm.",
        "",
        "Output requirements (repeat):",
        "- Output MUST be a single JSON object:",
        '  {\"shallow_llm\": bool, \"deep_web_llm\": bool, \"pdf_llm\": bool, \"img_llm\": bool}',
        "- Exactly one of these four booleans MUST be true. All three others MUST be false.",
        "- Do NOT add any extra keys, text, comments, or formatting.",
    ])


    format: JSONFormat = JSONFormat(
        answer = Answer( query = query,
            other_dict = [{
                'role': 'system',
                'content': system_prompt, 
            }]
        ),
        format = PathRecrute,
    )

    format = ollama_views.json_output(
        query = format,
        model = model,
        options=OllamaOptions(temperature=0)
    )


    return format




if __name__ == "__main__":
    while True:
        query = input('input>>> ')
        format = llm_recruter(query)
        print(format.output)




    





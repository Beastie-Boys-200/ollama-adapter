from models.Answer import ( 
        Answer, RagAnswer, JSONFormat, 
        Conversation, ImageAnswer, ToolCall
)
from models.ollama import OllamaOptions
from controllers import ollama as controller
from pathlib import Path
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from rich.panel import Panel










def rag_answer(query: RagAnswer, model: str, separate_context: bool = True, history: list[Answer] | None = None,
               options: OllamaOptions | None = None) -> RagAnswer:
    if query.answer is not None:
        raise ValueError("Message are already answered")


    messages = list() if history is None else [i for ans in history for i in ans.answer_dict]
    messages += query.answer_dict(separate_context)

    response: dict[str, str] = controller.answer(
        messages = messages, 
        model = model, 
        options = options.get_dict if options is not None else options
    )

    query.set_answer(response)

    return query



def stream_rag_answer(query: RagAnswer, model: str, separate_context: bool = True, history: list[Answer] | None = None,
               options: OllamaOptions | None = None):
    if query.answer is not None:
        raise ValueError("Message are already answered")


    messages = list() if history is None else [i for ans in history for i in ans.answer_dict]
    messages += query.answer_dict(separate_context)

    for token in controller.stream_answer(
        messages = messages, 
        model = model, 
        options = options.get_dict if options is not None else options
    ):
        yield token





# answer for text/image question
def answer(query: Answer, model: str, history: list[Answer] | None = None, options: OllamaOptions | None = None) -> Answer:
    if query.answer is not None:
        raise ValueError("Message are already answered")

    messages = list() if history is None else [i for ans in history for i in  ans.answer_dict]
    messages += query.answer_dict

    response: dict[str, str] = controller.answer(
        messages = messages, 
        model = model,
        options = options.get_dict if options is not None else options
    )
    query.set_answer(response)

    
    return query


def stream_answer(query: Answer, model: str, history: list[Answer] | None = None, options: OllamaOptions | None = None):
    if query.answer is not None:
        raise ValueError("Message are already answered")

    messages = list() if history is None else [i for ans in history for i in  ans.answer_dict]
    messages += query.answer_dict

    for token in controller.stream_answer(
        messages = messages, 
        model = model,
        options = options.get_dict if options is not None else options
    ):
        yield token





def json_output(query: JSONFormat, model: str, options: OllamaOptions | None = None) -> JSONFormat:
    if query.output is not None:
        raise ValueError("Message are already answered")
    if isinstance(query.answer, RagAnswer):
        messages = query.answer.answer_dict(separate_context=False)
    else:
        messages = query.answer.answer_dict 


    query.output = controller.json_answer(
        messages=messages, 
        model=model, 
        format=query.format,
        options = options.get_dict if options is not None else options
    )
    return query
        



def get_embendings(texts: list[str], model: str) -> np.ndarray:
    result = [controller.get_embedding(text, model) for text in texts]
    return np.array(result)



def answer_with_tools(query: ToolCall, model: str, options: OllamaOptions | None = None) -> ToolCall:
    if query.tools_execution is not None:
        raise ValueError("Message are already answered")

    if isinstance(query.answer, RagAnswer):
        messages = query.answer.answer_dict(separate_context=False)
    else:
        messages = query.answer.answer_dict 

    result = controller.tool_calling(
        messages = messages,
        tools = query.get_tool_dict,
        model = model,
        options = options.get_dict if options is not None else options
    )


    query.answer.set_answer(result[-1])
    query.tools_execution = result[:-1]



    return query




# use for long conversation
def next_gen(conv: Conversation, history_top_k: int = 1) -> Conversation:
    conv.last_answer = answer(
        query = conv.last_answer, 
        model = conv.model, 
        history = conv.history[-history_top_k:]
    )
    return conv



def make_conv_with_rich(console, model: str = "gemma3:12b") -> None:
    #console = Console()
    conv = Conversation(model=model)
    history_length = 0
    
    while True:
        query: str = console.input("[green]Enter your query >>> [green]")
        if query == "/bye":
            break
        conv.last_answer = Answer(query=query)
        history_length += 1
        with console.status("Generating"):
            conv = next_gen(conv, history_length)
        answer = Markdown(conv.last_answer.answer if conv.last_answer.answer is not None else '')
        console.print(Panel(answer, title="Model response"))

    





if __name__ == "__main__":


    # -- testing stream call --
    for i in stream_answer(Answer(query='hello'), model='llama3.2:1b'):
        print(i, end='', flush=True)
    exit()

    # -- testing tool call --- 
    def add(a: int, b: int) -> int:
      """Add two numbers"""
      """
      Args:
        a: The first number
        b: The second number

      Returns:
        The sum of the two numbers
      """
      return a + b


    def multiply(a: int, b: int) -> int:
      """Multiply two numbers"""
      """
      Args:
        a: The first number
        b: The second number

      Returns:
        The product of the two numbers
      """
      return a * b

   
    test_tool = ToolCall(
        answer = Answer(query='What is (11434+12341)*412?'),
        tools = [add, multiply]
    )

    test_tool = answer_with_tools(test_tool, model="glm-4.6:cloud")
    print(test_tool)

    
    

    
    # --- previous testing ---
    model = "llama3.2:1b"

    print("Test text answer: \n")
    ans = Answer(query="Hello, please tell me quick fun story")
    ans = answer(ans, model)
    print(ans)

    print("\n\nTest rag answer 1: \n")
    context = ["My name is Konqi", "I am a KDE mascot"]
    rag_ans = RagAnswer(query="What is my name?", context=context)
    rag_ans = rag_answer(rag_ans, model, separate_context=False)
    print(rag_ans)

    print("\n\nTest rag answer 2: \n")
    rag_ans = RagAnswer(query="Who i am?", context=context)
    rag_ans = rag_answer(rag_ans, model, separate_context=True)
    print(rag_ans)


    print("\n\nTest image answer: \n")
    response = requests.get('https://m.media-amazon.com/images/I/81yg-dRxBVL._UF1000,1000_QL80_.jpg')
    img = Image.open(BytesIO(response.content)).tobytes()
    img_ans = ImageAnswer(query="What are you see on image", paths=[img])
    img_ans = ImageAnswer(query="What are you see on image", paths=[Path("/home/linux/Desktop/gotou.jpg")])
    img_ans = answer(img_ans, model)
    print(img_ans)

    print("\n\n Test Json answer: ")
    class Format(BaseModel):
        city: str
        country: str
        language: str

    json_format = JSONFormat(
        answer = Answer(query="Basic info about kosice"),
        format=Format
    )
    json_format = json_output(json_format, model)
    print(json_format.answer.query)
    print(json_format.output)

    print("\n\n Test Json rag answer : ")
    class Format(BaseModel):
        name: str
        possition: str

    json_format = JSONFormat(
        answer = rag_ans,
        format=Format
    )
    json_format = json_output(json_format, model)
    print(json_format.answer.query)
    print(json_format.output)



    #print("\n\n Test Json image answer : ")
    #class Format(BaseModel):
    #    hair_color: str
    #    short_desc: str
    #    anime_character: bool

    #json_format = JSONFormat(
    #    answer = img_ans,
    #    format=Format
    #)
    #json_format = json_output(json_format, model)
    #print(json_format.answer.query)
    #print(json_format.output)


    try:
        make_conv_with_rich()
    except Exception:
        print("End of conversation")
        pass





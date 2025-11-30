#import ollama
import os
from ollama import Client
from ollama._types import ChatResponse
from pathlib import Path
from pydantic import BaseModel
import numpy as np
from typing import Any, Callable


print("ENVIRON", os.environ.get("OLLAMA_HOST", "localhost:11434"))

ollama = Client(
    host = os.environ.get("OLLAMA_HOST", "localhost:11434")
)


def answer(messages: list[dict[str, str]], model: str, options: dict[str, Any] | None) -> dict[str, str]:
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options
        
    response: ChatResponse = ollama.chat(model, messages, **kwargs)
    output = { 
        'role': response.message.role,
        'content': response.message.content
    }
    
    if response.message.thinking: output['thinking'] = response.message.thinking

    return output




def stream_answer(messages: list[dict[str, str]], model: str, options: dict[str, Any] | None):
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options
    # parameter to ollama for set stream
    kwargs['stream'] = True
        
    for token in ollama.chat(model, messages=messages, **kwargs):
        yield token['message']['content']
            
    





def json_answer(messages: list[dict[str, str]], model: str, format: type[BaseModel], options: dict[str, Any] | None) -> BaseModel:
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options

    response: ChatResponse = ollama.chat(
        messages= messages,
        model=model,
        format=format.model_json_schema(),
        **kwargs
    )

    if response.message.content is None:
        raise ValueError("Error when generating structure output message")



    return format.model_validate_json(response.message.content)


def get_embedding(text: str, model: str) -> np.ndarray:
    result = ollama.embed(model, text)

    # 1) Моделі типу all-minilm, nomic, mxbai -> embeddings=[[vector]]
    if "embeddings" in result:
        emb = result["embeddings"]

        if isinstance(emb, list) and len(emb) > 0:
            vec = emb[0]

            # Перевірка, що це саме 1D-вектор
            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                return np.array(vec, dtype=float)

    # 2) Моделі, які повертають "embedding": [...]
    if "embedding" in result:
        vec = result["embedding"]
        if isinstance(vec, list):
            return np.array(vec, dtype=float)

    # Якщо щось пішло не так — повертаємо None або кидаємо помилку
    raise ValueError(
        f"Model '{model}' did not return valid embeddings. Raw response: {result}"
    )




def tool_calling(messages: list[dict[str, str]], tools: dict[str, Callable], model: str, options: dict[str, Any] | None) -> list[dict[str, str]]:
    kwargs = dict() 
    if options is not None:
        kwargs['options'] = options
    
    tools_func: list[Callable] = [tools[i] for i in tools]

    messages_start_length = len(messages)

    while True:
        response: ChatResponse = ollama.chat(
            messages= messages,
            model=model,
            tools = tools_func, 
            **kwargs
        )
        messages.append({
            'role': response.message.role,
            'content': response.message.content
        })
        if response.message.thinking: messages[-1]['thinking'] = response.message.thinking

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                if tc.function.name in tools:
                    result = tools[tc.function.name](**tc.function.arguments)
                    messages.append({
                        'role': 'tool', 
                        'tool_name': tc.function.name, 
                        'content': str(result)
                    })

        else: break


    return messages[messages_start_length:]





if __name__ == "__main__":
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



    available_functions = {
      'add': add,
      'multiply': multiply,
    }


    messages = [{'role': 'user', 'content': 'What is (11434+12341)*412?'}]
    tool_result = tool_calling(messages, tools=available_functions, model="glm-4.6:cloud", options=None)
    print(tool_result)

    class Result(BaseModel):
        result: int

    print(json_answer(messages+tool_result, format=Result, model="llama3.2:1b", options=None))



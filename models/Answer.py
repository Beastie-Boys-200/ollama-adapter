from pydantic import BaseModel, ConfigDict
from enum import Enum
from pathlib import Path
from io import BytesIO
from typing import Callable 


# basic model for answer
class Answer(BaseModel):
    query: str # query to llm
    query_role: str = 'user' # role for query
    thinking: str | None = None # thinking part from llm response
    answer: str | None = None # llm response
    model_role: str | None = None # llm role
    # other dict can be specify in answer, for put it to model
    other_dict: list[dict[str, str]] | None = None


    def set_answer(self, response: dict[str, str]) -> None:
        self.model_role = response['role']
        self.answer = response['content']
        if 'thinking' in response:
            self.thinking = response['thinking']



    @property
    def answer_dict(self) -> list[dict[str, str]]:
        result = self.other_dict if self.other_dict else []
        result += [{
            'role': self.query_role,
            'content': self.query
        }]
        if self.answer is not None:
            result += [{
                'role': self.model_role,
                'content': self.answer
            }]

        return result

    
    def __str__(self) -> str:
        return f"Query: {self.query}\nAnswer: {self.answer}"



# model for raganswer with context
class RagAnswer(Answer):
    context: list[str]



    def answer_dict(self, separate_context: bool) -> list[dict[str, str]]:
        messages = self.other_dict if self.other_dict else []

        if separate_context:
            messages += [
                {
                    'role': 'user',
                    'content': "\n".join(self.context),
                },
                {
                    'role': 'user',
                    'content': self.query,  
                },
        ]
        else:
            messages.append({
                'role': 'user',
                'content': str((
                    f'Query: {self.query}',
                    "\n\n\n\n\n",
                    "Context: \n",
                    "\n".join(self.context)
                ))
            })
        return messages



# model for answer with image
class ImageAnswer(Answer):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    paths: list[str] | list[Path] | list[bytes]

    @property
    def answer_dict(self):
        messages = super().answer_dict
        if not len(self.paths):
            return messages
        elif isinstance(self.paths[0], Path):
            for path in self.paths:
                if not path.exists():
                    raise ValueError(f"Image: {path} does not exist")
            messages[0]['images'] = self.paths 
        else:
            messages[0]['images'] = self.paths
        return messages



class JSONFormat(BaseModel):
    answer: Answer
    format: type[BaseModel]
    output: BaseModel | None = None



class ToolCall(BaseModel):
    answer: Answer
    tools: list[Callable]
    tools_execution: list[dict[str, str]] | None = None

    @property
    def get_tool_dict(self) -> dict[str, Callable]:
        return { i.__name__: i for i in self.tools}






class Provider(Enum):
    ollama = "ollama"


# model for conversation
class Conversation(BaseModel):
    
    history: list[Answer] = []
    model: str
    provider: Provider = Provider.ollama



    @property
    def last_answer(self) -> Answer:
        return self.history[-1]

    @last_answer.setter
    def last_answer(self, answer: Answer):
        self.history.append(answer)


    def __str__(self) -> str:
        return f'Model: {self.provider}:{self.model} with history: {len(self.history)}'












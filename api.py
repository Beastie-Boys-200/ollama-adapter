import os
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from .views import ollama as ollama_provider
from .models.Answer import *
from .models.ollama import OllamaOptions

app = FastAPI(
    title="LLM Providers",
)

DEFAULT_OLLAMA_MODEL:str = os.environ.get('DEFAULT_OLLAMA_MODEL', 'llama3.2:1b')
DEFAULT_OLLAMA_EMB_MODEL: str = os.environ.get('DEFAULT_OLLAMA_EMB_MODEL', 'all-minilm:22m')
DEFAULT_OLLAMA_IMG_MODEL: str = os.environ.get('DEFAULT_OLLAMA_IMG_MODEL', 'moondream:1.8b')

# --- TEXT ENDPOINTS --- 

@app.post('/ollama/text/answer', tags=['text'])
def text_answer(query: Answer, model: str| None = None, opt: OllamaOptions | None = None) -> Answer:
    return ollama_provider.answer(
        query=query,
        model = model or DEFAULT_OLLAMA_MODEL,
        options=opt
    )


@app.get('/ollama/text/answer', tags=['text'])
def get_text_answer(query: str, model: str | None = None) -> Answer:
    opt = OllamaOptions(numa=True, num_gpu=2, main_gpu=0)
    return text_answer(Answer(query=query), model, opt) 



@app.post('/ollama/text/answer/stream', tags=['text-stream'])
def text_answer_stream(query: Answer, model: str| None = None, opt: OllamaOptions | None = None):
    return StreamingResponse(ollama_provider.stream_answer(
        query=query,
        model = model or DEFAULT_OLLAMA_MODEL,
        options=opt
    ), media_type='text')


@app.get('/ollama/text/answer/stream', tags=['text-stream'])
def get_text_answer_stream(query: str, model: str| None = None):
    return StreamingResponse(ollama_provider.stream_answer(
        query=Answer(query=query),
        model = model or DEFAULT_OLLAMA_MODEL,
    ), media_type='text')


@app.post('/ollama/text/raganswer', tags=['RAG'])
def text_raganswer(query: RagAnswer, model: str | None = None, opt: OllamaOptions | None = None) -> RagAnswer:
    return ollama_provider.rag_answer(
        query = query,
        model = model or DEFAULT_OLLAMA_MODEL,
        options=opt
    )


@app.post('/ollama/text/raganswer/stream', tags=['RAG-stream'])
def stream_text_raganswer(query: RagAnswer, model: str | None = None, opt: OllamaOptions | None = None):
    return StreamingResponse(ollama_provider.stream_rag_answer(
        query = query,
        model = model or DEFAULT_OLLAMA_MODEL,
        options=opt
    ), media_type='text')



# --- EMBENDDINGS ENDPOINTS --- 

@app.post('/ollama/text/embenddings', tags=['text-embendding'])
def text_embenddings(texts:list[str], model: str | None = None) -> list[list[float]]:
    return ollama_provider.get_embendings(
        texts=texts, 
        model = model or DEFAULT_OLLAMA_EMB_MODEL
    ).tolist()



# --- IMAGE ENDPOINTS ---

@app.post('/ollama/image/answer', tags=['images'])
def image_answer_by_url(query: str, urls: list[str], model: str | None = None) -> Answer:
    imgs_b: list[bytes] = []
    for url in urls:
        resp = requests.get(url)
        resp.raise_for_status()
        imgs_b.append(resp.content)

    answer = ollama_provider.answer(
        query = ImageAnswer(
            query=query,
            paths=imgs_b
        ),
        model=model or DEFAULT_OLLAMA_IMG_MODEL
    )

    answer.paths = urls
    return answer




@app.post('/ollama/image/image-answer/', tags=['images'])
def image_answer_by_imageanswer_with_url(query: ImageAnswer, model: str | None = None) -> Answer:
    imgs_b: list[bytes] = []
    for url in query.paths:
        resp = requests.get(str(url))
        resp.raise_for_status()
        imgs_b.append(resp.content)
    
    urls = query.paths
    query.paths = imgs_b

    answer = ollama_provider.answer(
        query = query,
        model=model or DEFAULT_OLLAMA_IMG_MODEL
    )

    answer.paths = urls
    return answer



@app.post('/ollama/image/answer/stream', tags=['images-stream'])
def stream_image_answer_by_url(query: str, urls: list[str], model: str | None = None):
    imgs_b: list[bytes] = []
    for url in urls:
        resp = requests.get(url)
        resp.raise_for_status()
        imgs_b.append(resp.content)

    return StreamingResponse(ollama_provider.stream_answer(
        query = ImageAnswer(
            query=query,
            paths=imgs_b
        ),
        model=model or DEFAULT_OLLAMA_IMG_MODEL
    ), media_type='text')




@app.post('/ollama/image/image-answer/stream', tags=['images-stream'])
def stream_image_answer_by_imageanswer_with_url(query: ImageAnswer, model: str | None = None):
    imgs_b: list[bytes] = []
    for url in query.paths:
        resp = requests.get(url)
        resp.raise_for_status()
        imgs_b.append(resp.content)

    query.paths = imgs_b

    return StreamingResponse(ollama_provider.stream_answer(
        query = query,
        model=model or DEFAULT_OLLAMA_IMG_MODEL
    ), media_type='text')





from . import ollama as ollama_views
from models.Answer import *
from models.ollama import OllamaOptions
from pydantic import BaseModel 
from controllers import pdf_reader 
import requests
import numpy as np


FAISS_URL = "http://localhost:8004"


def docs_pipeline(query: str, collection_name: str, docs_path: list[Path] | None = None):

    # read docs if provided
    if docs_path is not None: 
        chunks: list[str] = []

        for doc in docs_path:
            chunks += pdf_reader.read_pdf(doc)


        # make vectors from text
        embedding = ollama_views.get_embendings(chunks, model='embeddinggemma')

        # update vector db if docs provided
        if collection_name not in requests.get(f"{FAISS_URL}/faiss/collections/").json():
            requests.post(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json = {
                    "vectors": embedding.tolist(),
                    "metadata": {
                        "text": chunks
                    }
                }
            )
        else:
            requests.put(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json = {
                    "vectors": embedding.tolist(),
                    "metadata": {
                        "text": chunks
                    }
                }
            )
            

    # get embendings from query
    query_emb = ollama_views.get_embendings([query], model='embeddinggemma')[-1]

    # search top k simple query
    similar = requests.post(
        f"{FAISS_URL}/faiss/collections/{collection_name}/similar",
        json=query_emb.tolist()
    ).json()[-1]


    # generate answer on it
    return ollama_views.stream_rag_answer(
        query=RagAnswer(
            query=query,
            context=[vocab['text'] for vocab in similar]
        ),
        model='llama3:latest'
    )



def image_pipeline(query: str, collection_name: str, images_path: list[Path] | None = None):

    # describe images if they provided
    if images_path is not None:
        images_disc = []
        for image in images_path:
            images_disc.append(
                ollama_views.answer(
                    query=ImageAnswer(
                        query="Please provide full describe and information for this image",
                        paths=[image]
                    ),
                    model="gemma3:27b"
                ).answer
            )

        # make vectors from images descriptions
        img_embedding = ollama_views.get_embendings(images_disc, model='embeddinggemma')

        # update vector db if images provided
        if collection_name not in requests.get(f"{FAISS_URL}/faiss/collections/").json():
            requests.post(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json = {
                    "vectors": img_embedding.tolist(),
                    "metadata": {
                        "text": images_disc 
                    }
                }
            )
        else:
            requests.put(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json = {
                    "vectors": img_embedding.tolist(),
                    "metadata": {
                        "text": images_disc 
                    }
                }
            )

    
    # get embendings from query
    query_emb = ollama_views.get_embendings([query], model='embeddinggemma')[-1]

    # search for most similar text chunks
    similar = requests.post(
        f"{FAISS_URL}/faiss/collections/{collection_name}/similar",
        json=query_emb.tolist()
    ).json()[-1]


    # stream final answer with similar context
    return ollama_views.stream_rag_answer(
        query=RagAnswer(
            query=query,
            context=[vocab['text'] for vocab in similar]
        ),
        model='llama3:latest'
    )
    





    












if __name__ == "__main__":


        #for token in docs_pipeline(
        #    query="What you know about docker",
        #    collection_name="devops default",
        #    docs_path=[Path('./pdf_samples/devops1.pdf')]
        #):
        #    print(token, end=" ", flush=True)


    for token in image_pipeline(
        query="What "
    )











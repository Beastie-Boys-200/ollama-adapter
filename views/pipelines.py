from . import ollama as ollama_views
from models.Answer import *
from controllers import pdf_reader
import requests
from .scraper import search_and_extract
from .clean import semantic_clean
from .llm_planer import validate_with_metadata
from .llm_router import llm_router
from .planer import llm_planner

FAISS_URL = "http://localhost:8004"


def docs_pipeline(
    query: str, collection_name: str, docs_path: list[Path] | list[bytes] | None = None
):

    # read docs if provided
    if docs_path is not None:
        chunks: list[str] = []

        for doc in docs_path:
            chunks += pdf_reader.read_pdf(doc)

        # make vectors from text
        embedding = ollama_views.get_embendings(chunks, model="embeddinggemma")

        # update vector db if docs provided
        if (
            collection_name
            not in requests.get(f"{FAISS_URL}/faiss/collections/").json()
        ):
            requests.post(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json={"vectors": embedding.tolist(), "metadata": {"text": chunks}},
            )
        else:
            requests.put(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json={"vectors": embedding.tolist(), "metadata": {"text": chunks}},
            )

    # get embendings from query
    query_emb = ollama_views.get_embendings([query], model="embeddinggemma")[-1]

    # search top k simple query
    similar = requests.post(
        f"{FAISS_URL}/faiss/collections/{collection_name}/similar",
        json=query_emb.tolist(),
    ).json()[-1]

    # generate answer on it
    return ollama_views.stream_rag_answer(
        query=RagAnswer(
            query=query,
            context=[vocab["text"] for vocab in similar],
            other_dict=[
                {
                    "role": "system",
                    "content": "If provided information does not support with user question, simply answer that you can not answer to this query with provided context",
                }
            ],
        ),
        model="llama3:latest",
    )


def image_pipeline(
    query: str, collection_name: str, images_path: list[Path] | list[bytes] | None = None
):

    # describe images if they provided
    if images_path is not None:
        images_disc = []
        for image in images_path:
            images_disc.append(
                ollama_views.answer(
                    query=ImageAnswer(
                        query="Please provide full describe and information for this image",
                        paths=[image],
                    ),
                    model="gemma3:27b",
                ).answer
            )

        # make vectors from images descriptions
        img_embedding = ollama_views.get_embendings(images_disc, model="embeddinggemma")
        print(images_disc)

        # update vector db if images provided
        if (
            collection_name
            not in requests.get(f"{FAISS_URL}/faiss/collections/").json()
        ):
            requests.post(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json={
                    "vectors": img_embedding.tolist(),
                    "metadata": {"text": images_disc},
                },
            )
        else:
            requests.put(
                f"{FAISS_URL}/faiss/collection/{collection_name}",
                json={
                    "vectors": img_embedding.tolist(),
                    "metadata": {"text": images_disc},
                },
            )

    # get embendings from query
    query_emb = ollama_views.get_embendings([query], model="embeddinggemma")[-1]

    # search for most similar text chunks
    similar = requests.post(
        f"{FAISS_URL}/faiss/collections/{collection_name}/similar",
        json=query_emb.tolist(),
    ).json()[-1]

    # stream final answer with similar context
    return ollama_views.stream_rag_answer(
        query=RagAnswer(query=query, context=[vocab["text"] for vocab in similar]),
        model="llama3:latest",
    )


def web_search_pipeline(query: str, count: int, collection_name: str, list_of_query: list[str]):
    #raw_texts = search_and_extract(query, count)
    #texts = semantic_clean([text["text"] for text in raw_texts], with_log=False)

    texts = []
    for web_query in list_of_query:
        raw_texts = search_and_extract(web_query, count)
        texts += semantic_clean([ text["text"] for text in raw_texts ], with_log=False)


    # --- split web chunks for provide smaller chunks ---
    texts = [ text for text in pdf_reader.chunker(texts) if len(text) > 70 ]

    # make vectors from text
    embedding = ollama_views.get_embendings(texts, model="embeddinggemma")

    # update vector db if docs provided
    if (
        collection_name
        not in requests.get(f"{FAISS_URL}/faiss/collections/").json()
    ):
        requests.post(
            f"{FAISS_URL}/faiss/collection/{collection_name}",
            json={"vectors": embedding.tolist(), "metadata": {"text": texts}},
        )
    else:
        requests.put(
            f"{FAISS_URL}/faiss/collection/{collection_name}",
            json={"vectors": embedding.tolist(), "metadata": {"text": texts}},
        )


    # get embendings from query
    query_emb = ollama_views.get_embendings([query], model="embeddinggemma")[-1]

    # search top k simple query
    similar = requests.post(
        f"{FAISS_URL}/faiss/collections/{collection_name}/similar",
        json=query_emb.tolist(),
    ).json()[-1]

    # generate answer on it
    return ollama_views.stream_rag_answer(
        query=RagAnswer(
            query=query,
            context=[vocab["text"] for vocab in similar],
            other_dict=[
                {
                    "role": "system",
                    "content": "If provided information does not support with user question, simply answer that you can not answer to this query with provided context. Please provide deep answer with sources from provided contex.",
                }
            ],
        ),
        model="llama3:latest",
    )




def main_pipeline(query: str, doc: bytes | None = None, img: bytes | None = None, conversation_id: str = '123123'):
    doc_flag, img_flag = doc is not None, img is not None

    # --- first and second agent stages --- 
    result = validate_with_metadata(query, has_image=img_flag, has_doc=doc_flag)

    meaningful = result["meaningful"]  # Validation
    routing_validation = result["routing"]  # Validation | None
    
    # --- add instead continue LLM response ---

    if not meaningful.state:
        print(meaningful.text)
        raise ValueError("First agentic validation error")

    if not routing_validation.state:
        print(routing_validation.text)
        raise ValueError("Second agentic validation error")

    # -----------------------------------------

    
    # --- end of first and second agent stages --- 



    # llm router
    route = -1
    if img_flag:
        route = 3
    elif doc_flag:
        route = 2
    else:
        route = int(llm_router(query).output.route)


    # --- stream plan ---

    for token in llm_planner(query, route):
        #print(token, end="", flush=True)
        yield token

    # --- end of stream plan ---


    # --- get context from nikita service ---

    # ---------------------------------------


    # --- chose pipeline --- 

    if route == 0:
        # --- shallow model --- 
        for token in ollama_views.stream_answer(
            query = Answer(
                query=query,
            ), 
            model="llama3:latest"
        ): 
            #print(token, end="", flush=True)
            yield token

    elif route == 1:
        # --- web search ---
        class CreatedQuery(BaseModel):
            list_of_query: list[str]

        list_of_query = ollama_views.json_output(
            query = JSONFormat(
                answer=Answer(
                    query=query,
                    other_dict = [{
                        'role': 'system',
                        'content': " ".join([
                            "Please generate list of query input that will be use with web search",
                            "for search relevent information for user query"
                        ])
                    }]
                ), 
                format=CreatedQuery,
            ),
            model = 'llama3:latest'
        ).output.list_of_query

        
        for token in web_search_pipeline(query, 5, collection_name="web-parsing", list_of_query=list_of_query):
            #print(token, end="", flush=True)
            yield token

    elif route == 2:
        # --- documenet pipeline ---
        for token in docs_pipeline(
            query=query,
            collection_name=conversation_id,
            docs_path=[doc] if doc is not None else None,
        ):
            #print(token, end=" ", flush=True)
            yield token

    elif route == 3:
        # --- image pipeline ---
        for token in image_pipeline(
           query=query,
           collection_name=conversation_id,
           images_path = [img] if img is not None else None
        ):
           #print(token, end='', flush=True)
            yield token

    else:
        raise ValueError("Route is not match with our pipeline")




    
    


if __name__ == "__main__":

    # for token in docs_pipeline(
    #     query="What you know about Docker",
    #     collection_name="devops default",
    #     docs_path=[Path("./pdf_samples/devops1.pdf")],
    # ):
    #     print(token, end=" ", flush=True)

    # for token in image_pipeline(
    #    query="In what color monoliza was pained",
    #    #query="I provided you picture that represent avenue on paris, how this picture name and who is author",
    #    #query="What color pallet use picture that i am you send earlier, that have effel tower on it",
    #    collection_name="image_collection",
    #    #images_path = [
    #    #    Path('./image_samples/monoLiza.jpeg'),
    #    #    Path('./image_samples/picasso.jpg'),
    #    #    Path('./image_samples/sample1.jpg')
    #    #]
    # ):
    #    print(token, end='', flush=True)

    #for token in web_search_pipeline("Is Nintendo is originally company from Japan", 5, collection_name="web-parsing"):
    #    print(token, end="", flush=True)


    for token in main_pipeline("Please provide me fresh list with game station that Nintendo currently sell in Europe."):
        print(token, end="", flush=True)
    #main_pipeline("Hello")
    


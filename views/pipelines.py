from . import ollama as ollama_views
from models.Answer import *
from controllers import pdf_reader
import requests
from .scraper import search_and_extract
from .clean import semantic_clean
from .llm_planer import validate_with_metadata
from .llm_router import llm_router
from .planer import llm_planner
import json
import io
import base64

#FAISS_URL = "http://host.docker.internal:8004"
FAISS_URL = 'http://localhost:8004'


def docs_pipeline(
    query: str, collection_name: str, docs_path: list[Path] | list[bytes] | None = None
):

    # read docs if provided
    if docs_path is not None:
        chunks: list[str] = []

        for doc in docs_path:
            chunks += pdf_reader.read_pdf(doc)

        # make vectors from text
        embedding = ollama_views.get_embendings(chunks, model="all-minilm")

        # update vector db if docs provided
        if (
            collection_name
            not in requests.get(f"{FAISS_URL}/faiss/collections").json()
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
    query_emb = ollama_views.get_embendings([query], model="all-minilm")[-1]

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

        img_path = []
        for kika in images_path:
            # kika может быть bytes (data URL или base64) или str
            if isinstance(kika, bytes):
                data_url = kika.decode("utf-8")
            else:
                data_url = kika

            # если приходит вида "data:image/png;base64,AAAA..."
            if data_url.startswith("data:"):
                img_b64 = data_url.split(",", 1)[1]
            else:
                img_b64 = data_url  # уже чистый base64

            buffer = base64.b64decode(img_b64)  # это bytes с картинкой
            img_path.append(buffer)             # кладём СЫРЫЕ байты, не BytesIO



        
        images_path = img_path

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
        img_embedding = ollama_views.get_embendings(images_disc, model="all-minilm")

        # update vector db if images provided
        if (
            collection_name
            not in requests.get(f"{FAISS_URL}/faiss/collections").json()
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
    query_emb = ollama_views.get_embendings([query], model="all-minilm")[-1]

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



def web_search_pipeline(
    query: str, count: int, collection_name: str, list_of_query: list[str]
):
    # raw_texts = search_and_extract(query, count)
    # texts = semantic_clean([text["text"] for text in raw_texts], with_log=True)

    texts = []
    for web_query in list_of_query[:3]:
        raw_texts = search_and_extract(web_query, count)

        raw_text = [
            text["text"][idx : idx + 512]
            for text in raw_texts
            for idx in range(0, len(text["text"]), 512)
            if text["text"]
        ]

        # texts += semantic_clean([ text["text"] for text in raw_texts ], with_log=False)
        texts += semantic_clean(raw_text, with_log=False)

    # --- split web chunks for provide smaller chunks ---
    split_texts = []

    # 1) texts → список рядків
    for text in texts:
        # chunker повертає списки чанків
        chunks = pdf_reader.chunker(text)

        for chunk in chunks:
            # chunk може бути списком або рядком → приводимо до str
            if not isinstance(chunk, str):
                chunk = " ".join(chunk)

            # пропустити дрібні шматки
            if len(chunk.strip()) < 70:
                continue

            split_texts.append(chunk.strip())

    texts = split_texts

    # print(texts)

    # make vectors from text
    embedding = ollama_views.get_embendings(texts, model="all-minilm")

    # update vector db if docs provided
    if collection_name not in requests.get(f"{FAISS_URL}/faiss/collections").json():
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
    query_emb = ollama_views.get_embendings([query], model="all-minilm")[-1]

    # search top k simple query
    print()
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




class QueryPipeline(BaseModel):
    query: str
    doc: bytes | None = None
    img: bytes | None = None
    conversation_id: str = '123123'

#def main_pipeline(query: str, doc: bytes | None = None, img: bytes | None = None, conversation_id: str = '123123'):
def main_pipeline(query: QueryPipeline):
    doc_flag, img_flag = query.doc is not None, query.img is not None

    # --- first and second agent stages --- 
    result = validate_with_metadata(query.query, has_image=img_flag, has_doc=doc_flag)

    meaningful = result["meaningful"]  # Validation
    routing_validation = result["routing"]  # Validation | None
    
    # --- add instead continue LLM response ---

    if not meaningful.state:
        for token in meaningful.text.split(" "):
            yield json.dumps({ 'role': 'bot', 'token': token + " "}) + "\n"
        
        return json.dumps({ 'role': 'bot', 'token': " "}) + "\n"

        #print(meaningful.text)
        #raise ValueError("First agentic validation error")

    if not routing_validation.state:
        for token in routing_validation.text.split(" "):
            yield json.dumps({ 'role': 'bot', 'token': token + " "}) + "\n"

        return json.dumps({ 'role': 'bot', 'token': " "}) + "\n"

            #yield token
        #print(routing_validation.text)
        #raise ValueError("Second agentic validation error")

    # -----------------------------------------

    
    # --- end of first and second agent stages --- 



    # llm router
    route = -1
    if img_flag:
        route = 3
    elif doc_flag:
        route = 2
    else:
        route = int(llm_router(query.query).output.route)


    # --- stream plan ---

    for token in llm_planner(query.query, route):
        #print(token, end="", flush=True)
        yield json.dumps({ 'role': 'plan', 'token': token }) + "\n"

        #yield token

    # --- end of stream plan ---


    # --- get context from nikita service ---

    # ---------------------------------------


    # --- chose pipeline --- 

    if route == 0:
        # --- shallow model --- 
        for token in ollama_views.stream_answer(
            query = Answer(
                query=query.query,
            ), 
            model="llama3:latest"
        ): 
            #print(token, end="", flush=True)
            yield json.dumps({ 'role': 'bot', 'token': token }) + "\n"
            #yield token

    elif route == 1:
        # --- web search ---
        class CreatedQuery(BaseModel):
            list_of_query: list[str]

        list_of_query = ollama_views.json_output(
            query = JSONFormat(
                answer=Answer(
                    query=query.query,
                    other_dict = [{
                        'role': 'system',
                        'content': " ".join([
                            "Please generate list of query input that will be use with web search",
                            "for search relevent information for user query, it will be search in duckduckgo",
                            "so provide simple and small query"
                        ])
                    }]
                ), 
                format=CreatedQuery,
            ),
            model = 'llama3:latest'
        ).output.list_of_query

        #print("\nList of queries", list_of_query)
        yield json.dumps({ 'role': 'web link', 'token': "\n".join([
            f" {idx}. **{query}**" for idx, query in enumerate(list_of_query)
        ])}) + "\n"

        #yield "\nList of queries" + " ".join(list_of_query)

        list_of_query = list_of_query[:3]
        
        for token in web_search_pipeline(query.query, 3, collection_name=query.conversation_id, list_of_query=list_of_query):
            #print(token, end="", flush=True)
            yield json.dumps({ 'role': 'bot', 'token': token }) + "\n"

            #yield token

    elif route == 2:
        # --- documenet pipeline ---
        for token in docs_pipeline(
            query=query.query,
            collection_name=query.conversation_id,
            docs_path=[query.doc] if query.doc is not None else None,
        ):
            #print(token, end=" ", flush=True)
            yield json.dumps({ 'role': 'bot', 'token': token }) + "\n"

            #yield token

    elif route == 3:
        # --- image pipeline ---
        for token in image_pipeline(
           query=query.query,
           collection_name=query.conversation_id,
           images_path = [query.img] if query.img is not None else None
        ):
           #print(token, end='', flush=True)
            yield json.dumps({ 'role': 'bot', 'token': token }) + "\n"

            #yield token

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


    #for token in main_pipeline(QueryPipeline(query="What game console currently Nintendo sell in Europe? Could you please search freash information on web for provide this answer")):
        #print(token, end="", flush=True)

    for token in main_pipeline(QueryPipeline(query="What services provide Deutche Telekom in Slovakia, please provide me answer with deep research")):
        print(token, end="", flush=True)
    #main_pipeline("Hello")
    


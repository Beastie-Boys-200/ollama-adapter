FROM python:3.12.12-slim

EXPOSE 8000

WORKDIR /llm_providers/


ADD controllers/ ./controllers/.
ADD models/ ./models/.
ADD views/ ./views/.
ADD api.py ./.
ADD __init__.py ./.



ENV OLLAMA_HOST=http://host.docker.internal:11434

RUN pip install -U pip

COPY requirements.in .
RUN pip install -r requirements.in

#COPY requirements.txt .
#RUN pip install -r requirements.txt



CMD ["python3", "-m", "fastapi", "dev", "--host", "0.0.0.0", "api.py"]


FROM python:3.12.12-slim

EXPOSE 8000

WORKDIR /modules/


ADD controllers/* ./llm_providers/controllers/.
ADD models/* ./llm_providers/models/.
ADD views/* ./llm_providers/views/.
ADD api.py ./llm_providers/.
ADD __init__.py ./llm_providers/.

COPY ../__init__.py ./.


ENV OLLAMA_HOST=http://host.docker.internal:11434

RUN pip install -U pip

COPY requirements.in .
RUN pip install -r requirements.in

#COPY requirements.txt .
#RUN pip install -r requirements.txt



CMD ["python3", "-m", "fastapi", "dev", "--host", "0.0.0.0", "llm_providers/api.py"]


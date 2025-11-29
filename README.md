# LLM PROVIDERS module

Module for use llm with different model providers.

- ollama(**supported**)
- hugginface(**not supported**)
- vllm(**not supported**)
- Open AI API(**not supported**)
- Gemini API(**not supported**)
- Grock API(**not supported**)



# Image info

Normaly build Dockefile for create image. You have `OLLAMA_HOST` env variable, that specify ollama client url. If you want use ollama that run localy in your system, following next steps, default value of `OLLAMA_HOST` variable already set in image.

- Make sure that ollama run in 0.0.0.0 host ip, and can be accessably not only from local network
    add this `Environment="OLLAMA_HOST=0.0.0.0:11434"` in file `/etc/systemd/system/ollama.service` to `[Service]` section. 
- Run container with flag `--add-host=host.docker.internal:host-gateway`


### Run example
```sh
docker run --rm -it \
    -p 8000:8000 \
    --add-host=host.docker.internal:host-gateway \
    <llm-providers-image>
```







### Ollama setup
**Setup localy:** [ollama download page](https://ollama.com/download)
**Setup with docker container:** [docker hub](https://hub.docker.com/r/ollama/ollama)




##### Usefull links
- [ollama models catalog page](https://ollama.com/search)
- [ollama documentation](https://docs.ollama.com/)

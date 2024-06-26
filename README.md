# RAGTune
RAGTune is an application for improving _RAG_ performance by generating efficient _prompt tuning_ and _model finetuning_

![Image of chat interface](https://github.com/VivekSil/RAGTune/blob/main/images/chat_marked.png)

## Features
RAGTune support the following features
- Local chat: Completely local chat with [Ollama](https://www.ollama.com/) and [Weaviate](https://weaviate.io/)'s text2vec-transformers model inference
- Prompt tuning : Efficiently optimized prompt based on the instruction using [DSPy](https://github.com/stanfordnlp/dspy)
- Embedding finetuning: Finetuning embedding models on custom data using [LlamaIndex](https://www.llamaindex.ai/) and [Groq](https://groq.com/)

## Setup
This application requires `python==3.9`, `ollama` and `node`
- Backend
    - Run `cd backend `
    - Run `pip install -r requirements.txt`
    - Run `docker run -p 8080:8080 cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1` to set up the transformer model
    - Run `uvicorn main:app`

- Frontend
    - Run `cd frontend`
    - Run `npm install`
    - Run `npm start`
## Usage
Once the setup is complete, follow these steps to use RAGTune
- Local chat
    - Go to `localhost:3000`
    - Upload a text/PDF file
    - You can now chat
- Prompt tuning
    - Add instuction in the prompt tuning
    - Click on `Load` button once the optimization is done
- Embedding finetuning
    - Add the model to finetune and click 'Submit'

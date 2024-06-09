# RAGTune
RAGTune is an application for improving _RAG_ performance by generating efficient _prompt tuning_ and _model finetuning_

## Features
RAGTune support the following features
- Prompt tuning : Efficiently optimized prompt based on the instruction
- Embedding finetuning: Finetuning embedding models on custom data

## Setup
This application requires `python==3.9`, `ollama` and `node`
- Backend
    - Run `cd backend `
    - Run `pip install -r requirements.txt`
    - Run `docker-compose up` to set up the transformer model (uses [transformer-inference](https://hub.docker.com/r/semitechnologies/transformers-inference))
    - Run `uvicorn main:app`

- Frontend
    - Run `cd frontend`
    - Run `npm install`
    - Run `npm start`

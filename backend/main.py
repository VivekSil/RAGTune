import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
import os
import requests
import json
from dotenv import load_dotenv
import aiofiles
import socket
from pypdf import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rapt import *
from textdistance import cosine
from sentence_transformers import SentenceTransformer, util
from dspy.retrieve.weaviate_rm import WeaviateRM
from dspy.teleprompt import COPRO

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.groq import Groq
from llama_index.finetuning import SentenceTransformersFinetuneEngine

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO : Add multiple db options (Weaviate cloud and local)
# TODO : Add embedding fine-tuning and LLM finetuning support

client = weaviate.connect_to_embedded(
    environment_variables={"ENABLE_MODULES": "text2vec-transformers,generative-ollama", "TRANSFORMERS_INFERENCE_API":"http://localhost:8080"},
    additional_config=AdditionalConfig(timeout=Timeout(init=10000, query=10900, insert=15800)),
    version="1.25.1",
)
client.collections.delete_all()
collection = client.collections.create(
    "Research_papers",
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
    generative_config=wvc.config.Configure.Generative.ollama(
        api_endpoint="http://localhost:11434",
        model="phi3"
    )
)


load_dotenv()

lm = dspy.GROQ(model='llama3-8b-8192', api_key=os.environ["GROQ_API_KEY"])
retriever = WeaviateRM("Research_papers", weaviate_client=client)
dspy.settings.configure(rm=retriever, lm=lm)


def merge_lines(text_lines):
    chunks = []
    while text_lines:
        chunks.append('\n'.join(text_lines[:10]))
        if len(text_lines) <= 9:
            return chunks
        text_lines = text_lines[10:]
    return chunks   

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes

class OracleSignature(dspy.Signature):
    """You will be given a question and context. You need to answer the question with explanation based on the context given. If the answer doesn't lie in the context, say I don't know. Answer the given question after `Answer:` """

    question = dspy.InputField(desc="Question asked")
    context = dspy.InputField(desc="Potentially related passages")
    answer = dspy.OutputField(desc="Answer to the question based on the given context, just give answer, and nothing else")

class OracleRAFT(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_answer = dspy.ChainOfThought(OracleSignature)

    def forward(self, question):
        context = self.retrieve(question).toDict()["passages"]
        prediction = self.generate_answer(question=question, context=context)
        # return dspy.Prediction(context=context, answer=prediction.answer)
        return prediction

model = SentenceTransformer("all-MiniLM-L6-v2")
def embedding_sim_oracle(question, pred):
    question_embedding = model.encode(question.question)
    pred_embedding = model.encode(pred.rationale.split("Answer:")[-1])
    return float(util.dot_score(question_embedding, pred_embedding)[0][0])

additional_instruction = ""
global_filename = ""

@app.get("/api/embedding_finetuning")
async def embedding_finetuning(model:str):
    global global_filename
    text_nodes = load_corpus([global_filename])
    train_dataset = generate_qa_embedding_pairs(llm=Groq(model="llama3-8b-8192",api_key=os.environ["GROQ_API_KEY"]), nodes=text_nodes)
    train_dataset.save_json("train_dataset.json")
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,
        model_id=model,
        model_output_path=f"{model}-finetuned",
        val_dataset=train_dataset,
    )
    finetune_engine.finetune()
    return {"status":"Embedding finetuning complete"}

@app.get("/api/rapt_generate")
async def rapt_generate(query: str):
    global additional_instruction
    # Generate a response using a JSON object of schema {"response":response}. You will reply with only JSON and no explanation and other information
    sign = OracleSignature
    sign.__doc__+=additional_instruction
    program = OracleRAFT()
    program.generate_answer = dspy.ChainOfThought(sign)
    program.load("./ragcompiled.json")
    result = program(query).answer.split("Answer:")[-1]
    return {"response":result}

@app.get("/api/rapt_tune")
async def rapt_tune(text: str):
    global additional_instruction
    teleprompter_oracle = COPRO(metric=embedding_sim_oracle, depth=3, breadth=2)
    questions = ["Describe recurrent neural networks ?", "What is the subspaced similarity between different r in LoRA ?", "Describe scaled dot product attention"]
    answers = [
    "Recurrent Neural Networks (RNNs) are specialized neural networks designed for sequential data processing. They maintain a hidden state that captures information from past inputs, allowing them to exhibit dynamic temporal behavior. RNNs employ parameter sharing across time steps, enabling them to process sequences of varying lengths efficiently. Common applications include natural language processing, time series analysis, and speech recognition. Despite their effectiveness, RNNs struggle with capturing long-term dependencies due to vanishing or exploding gradients and are inefficient in parallel processing. Architectures like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) have been developed to address these limitations while retaining the core recurrent structure. RNNs remain fundamental in sequence modeling tasks but are often supplemented or replaced by more advanced architectures in scenarios requiring long-range dependencies and improved performance.",

    "In LoRA (Linearly-organized Recurrent Attention), the `subspaced similarity between different r` refers to a measure of similarity between different attention contexts (represented by `r`) within the model. LoRA introduces the concept of `subspaces` to the attention mechanism. Each attention context `r` is projected onto multiple subspaces, and the similarity between different attention contexts is measured within these subspaces. The subspaced similarity is computed using a dot product between the projected representations of attention contexts onto each subspace. By calculating similarity in multiple subspaces, LoRA allows for capturing diverse types of relationships between attention contexts, thereby enhancing the model's ability to capture nuanced dependencies in sequential data. This approach helps LoRA effectively model complex sequences by providing more flexibility and expressiveness in the attention mechanism.",

    "Scaled dot-product attention is a pivotal component of transformer models, facilitating effective capture of dependencies across input sequences. It operates by computing the dot product of query and key vectors, scaled to prevent vanishing gradients, followed by a softmax to obtain attention weights. These weights indicate the relevance of each value vector to its corresponding query. Ultimately, a weighted sum of the value vectors, weighted by the attention scores, produces the output. This attention mechanism enables the model to focus on pertinent information while processing input sequences, facilitating tasks such as machine translation, text generation, and language understanding. Its ability to capture long-range dependencies efficiently has contributed to the success of transformer-based architectures in various natural language processing applications."]
    trainset = [dspy.Example(question=questions[i], answer=answers[i]).with_inputs("question") for i in range(len(questions))]
    kwargs = dict(num_threads=1, display_progress=True, display_table=3)
    print("text:",text)
    additional_instruction = text
    sign = OracleSignature
    print("Docstring:",sign.__doc__)
    sign.__doc__+=text
    raft = OracleRAFT()
    raft.generate_answer = dspy.ChainOfThought(sign)
    compiled_oracle = teleprompter_oracle.compile(raft,trainset=trainset, eval_kwargs=kwargs)
    compiled_oracle.save("./ragcompiled.json")
    return {"status":"Program compiled successfully"}

@app.get("/api/generate")
async def generate(query: str):
    global client
    global collection
    collection = client.collections.get("Research_papers")
    result = collection.generate.near_text(query=query,limit=5,grouped_task=f"Answer the question: {query}")
    print(query)
    print(result)
    return {"response":result.generated}

@app.post("/api/file_upload")
async def upload_file(file: UploadFile = File(...)):
    global client
    global global_filename
    collection = client.collections.get("Research_papers")
    global_filename = file.filename

    if file.content_type == "application/pdf":
        async with aiofiles.open(file.filename, 'wb') as f:
            content = await file.read()
            await f.write(content)
        pdf_pages = PdfReader(file.filename).pages
        content = ''.join([page.extract_text() for page in pdf_pages])
    elif file.content_type == "text/plain":
        async with aiofiles.open(file.filename, "w") as f:
            content = await file.read()
            f.write(content)
            content = content.decode('utf-8')
    content = content.split("\n")
    documents = {}
    chunks = merge_lines(content)
    documents[file.filename] = chunks
    print(documents)
    # try:
    with collection.batch.dynamic() as batch:
        for doc in documents:
            for chunk in documents[doc]:
                batch.add_object(properties = {"content" : chunk, "source":doc})
    # except:
    #     return {"status":"Document not added"}

    return {"status":"Document added succesfully"}

@app.get("/api/get_client_info")
async def get_client_info():
    print(client.meta())
    return {"info":client.meta()}

@app.get("/api/close_client")
async def close_connection():
    global client
    client.close()

@app.get("/api/get_text")
def get_text(text: str):
    return text
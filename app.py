import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ----------------------------------
# App Init
# ----------------------------------

app = FastAPI(title="Local AI Backend")

VECTOR_DB_PATH = "vectorstore"

# ----------------------------------
# Load LLM (Ollama)
# ----------------------------------

llm = Ollama(
    model="gemma3:270m",
    temperature=0.2
)

# ----------------------------------
# Load Embeddings
# ----------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------------
# Initialize / Load Vector Store
# ----------------------------------

# if os.path.exists(VECTOR_DB_PATH):
#     vectorstore = FAISS.load_local(
#         VECTOR_DB_PATH,
#         embedding_model,
#         allow_dangerous_deserialization=True
#     )
# else:
#     # Sample knowledge base (replace with file ingestion later)
#     documents = [
#         Document(page_content="AWS Lambda supports timeouts up to 15 minutes."),
#         Document(page_content="S3 provides durable object storage."),
#         Document(page_content="IAM controls access to AWS services.")
#     ]

#     vectorstore = FAISS.from_documents(documents, embedding_model)
#     vectorstore.save_local(VECTOR_DB_PATH)

documents = [
    Document(page_content="AWS Lambda supports timeouts up to 15 minutes."),
    Document(page_content="S3 provides durable object storage."),
    Document(page_content="IAM controls access to AWS services.")
]

vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local(VECTOR_DB_PATH)


retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# ----------------------------------
# Request Model
# ----------------------------------

class QueryRequest(BaseModel):
    question: str

# ----------------------------------
# Routes
# ----------------------------------

@app.get("/")
def health_check():
    return {"status": "AI backend running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    response = qa_chain.invoke({"query": request.question})
    return {
        "question": request.question,
        "answer": response["result"]
    }
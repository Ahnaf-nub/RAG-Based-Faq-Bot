from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os

load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "faq-bot"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load FAQ data from the text file
data = pd.read_csv('faqs.txt', sep='\t', header=None, names=['question', 'answer'])

# Upsert data into Pinecone
questions = data['question'].tolist()
answers = data['answer'].tolist()
metadatas = [{'answer': ans} for ans in answers]
embeddings_list = embeddings.embed_documents(questions)

upsert_data = [
    (str(i), embeddings_list[i], metadatas[i])
    for i in range(len(questions))
]
index.upsert(vectors=upsert_data)

vectorstore = LangchainPinecone(index, embeddings.embed_query, text_key="answer")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.5
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/query")
async def query_faq(q: str):
    try:
        result = qa_chain.invoke({"query": q})
        answer = result['result']
        return {"response": answer}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
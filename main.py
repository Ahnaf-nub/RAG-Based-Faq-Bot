from dotenv import load_dotenv
import os
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

data = pd.DataFrame({
    "question": [
        "How can I reset my password?",
        "What is the return policy?",
        "How do I track my order?",
        "Can I cancel my subscription?",
        "What payment methods are accepted?",
        "How do I contact customer support?",
        "What are your shipping times?",
        "Do you ship internationally?",
        "How can I update my billing information?",
        "What are your business hours?",
        "Is there a warranty on products?",
        "How do I create an account?",
        "Where can I find size charts?",
        "Do you offer gift cards?",
        "What's your privacy policy?"
    ],
    "answer": [
        "To reset your password, click on 'Forgot Password' on the login page.",
        "Our return policy allows returns within 30 days of purchase.",
        "You can track your order using the tracking link sent to your email.",
        "To cancel your subscription, go to your account settings.",
        "We accept credit cards, debit cards, and PayPal.",
        "You can reach customer support via email at support@example.com or call 1-800-123-4567.",
        "Standard shipping takes 3-5 business days, express shipping is 1-2 business days.",
        "Yes, we ship to most countries worldwide. Shipping costs vary by location.",
        "Log into your account, go to 'Payment Methods' and click 'Edit'.",
        "We're open Monday-Friday, 9am-6pm EST.",
        "All products come with a 1-year limited warranty against manufacturing defects.",
        "Click 'Sign Up' at the top right of our homepage and fill out the form.",
        "Size charts can be found on each product page under 'Size Guide'.",
        "Gift cards are available for purchase in denominations from $25 to $500.",
        "Our privacy policy details how we collect and use your data. View it at example.com/privacy."
    ]
})

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
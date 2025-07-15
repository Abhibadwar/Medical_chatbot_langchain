import os
from dotenv import load_dotenv
import streamlit as st
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_huggingface_embeddings()
index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)
retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)

# Streamlit UI
st.set_page_config(page_title="MedicalBot Chat")
st.title("🩺 MedicalBot Chat Assistant")

user_input = st.chat_input("Ask me anything medical...")

if user_input:
    response = rag_chain.invoke({"input": user_input})
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response["answer"])

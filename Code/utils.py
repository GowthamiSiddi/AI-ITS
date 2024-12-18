from config import get_google_api_key, load_config
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings 
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
from langchain_groq import ChatGroq

load_config()
genai.configure(api_key='AIzaSyDuwNT8YYue3otvFVTI9g4PiOrJgPznr6Q')


# Function to read PDF text from a directory
def get_pdf_text_from_directory(directory_path="resourace"):
    text = ""
    pdf_files = glob.glob(f"{directory_path}/*.pdf")
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to chunk PDF text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to convert chunked text to embeddings and store locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("../faiss_index")

# Function to setup model, chain and included Prompt template
def get_conversational_chain():
    prompt_template = """
    You are an intelligent tutoring chatbot specialized in Python programming. 
    You are designed to assist users with Python-related questions only. 

    If the user's question is unrelated to Python, politely respond with: 
    "I am specialized in Python programming and currently do not have information on this topic."

    For Python-related questions, always provide clear, concise answers, include Python code examples with explanations, and clarify the outputs if applicable.

    Chat history: \n{chat_history}\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    chain = prompt | model
    
    return chain


def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")    

    # Load the FAISS index
    try:
        new_db = FAISS.load_local("../faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        docs = []

    # Setup conversational chain
    chain = get_conversational_chain() | StrOutputParser()

    # Provide context or fallback to general knowledge
    if docs:
        response = chain.invoke({"chat_history": chat_history, "context": docs, "question": user_question})
    else:
        # Use the chain for general knowledge without context
        response = chain.invoke({"chat_history": chat_history, "context": "None", "question": user_question})

    return response
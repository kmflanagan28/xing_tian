import streamlit as st
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CORE RAG SETUP FUNCTION ---
@st.cache_resource
def setup_rag_chain():
    """
    Sets up the Retrieval-Augmented Generation (RAG) chain.
    """
    # Create and set a new event loop for environments like Streamlit
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # LOAD: Load the knowledge base from the text file
    loader = TextLoader("knowledge.txt")
    docs = loader.load()

    # SPLIT: Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # EMBED & STORE: Create the vector store
    # The GoogleGenerativeAIEmbeddings function will automatically use the GOOGLE_API_KEY
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory="chroma_db"
    )

    # Set up the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    # Set up the Retriever
    retriever = vectorstore.as_retriever()

    # Define the prompt template
    prompt_template = """
    You are an expert assistant for Project Starlight.
    Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, state that you don't have enough information.
    Do not make up information.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# --- STREAMLIT UI ---
st.set_page_config(page_title="Project Starlight Assistant", page_icon="🌟")
st.title("🌟 Project Starlight Assistant")
st.write("Ask me anything about Project Starlight. I have access to the mission briefing.")

# Set up the RAG chain and handle potential errors during setup
try:
    rag_chain = setup_rag_chain()
except Exception as e:
    # If there's an error here, it's likely the API key is still invalid or missing
    st.error(f"An error occurred during RAG setup: {e}")
    st.info("This error often means the GOOGLE_API_KEY is not configured correctly. Please check your .env file and restart the app.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you with Project Starlight today?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

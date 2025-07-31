import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import signal # Required to stop the process

# Load environment variables from .env file
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader # Updated Loader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. DEFINE KNOWLEDGE BASE DIRECTORY ---
KNOWLEDGE_BASE_DIR = "knowledge_base"
HISTORY_DIR = os.path.join(KNOWLEDGE_BASE_DIR, "history")

# --- 2. CORE RAG SETUP FUNCTION ---
@st.cache_resource
def setup_rag_chain():
    """
    Sets up the Retrieval-Augmented Generation (RAG) chain.
    Now loads all .txt files from the knowledge_base directory.
    """
    # Create and set a new event loop for environments like Streamlit
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Ensure the knowledge base directory exists
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)

    # LOAD: Load all .txt files from the knowledge_base directory and its subdirectories
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", recursive=True)
    docs = loader.load()

    # If no documents are found, provide a message and stop
    if not docs:
        st.warning(f"No documents found in the '{KNOWLEDGE_BASE_DIR}' directory. Please add a .txt file (e.g., knowledge.txt) to it.")
        return None

    # SPLIT: Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # EMBED & STORE: Create the vector store
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
    You are an expert assistant for Project Starlight. You have access to a knowledge base and a history of past conversations.
    Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, state that you don't have enough information.

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

def save_conversation(prompt, response):
    """Saves the user prompt and AI response to a timestamped file."""
    # Ensure the history directory exists
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    
    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(HISTORY_DIR, f"conversation_{timestamp}.txt")
    
    # Write the conversation to the file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"User: {prompt}\n")
        f.write(f"Assistant: {response}\n")

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Project Starlight Assistant", page_icon="🌟")

# --- ADD A SIDEBAR WITH A STOP BUTTON ---
with st.sidebar:
    st.header("Controls")
    if st.button("Stop Application"):
        st.success("Stopping application... You can close this tab.")
        # Sends a signal to the process to terminate, effectively like Ctrl+C
        os.kill(os.getpid(), signal.SIGTERM)


st.title("🌟 Project Starlight Assistant")
st.write(f"My knowledge is based on the files in the '{KNOWLEDGE_BASE_DIR}' directory.")

# Set up the RAG chain and handle potential errors during setup
try:
    rag_chain = setup_rag_chain()
except Exception as e:
    st.error(f"An error occurred during RAG setup: {e}")
    st.info("This error often means the GOOGLE_API_KEY is not configured correctly. Please check your .env file and restart the app.")
    st.stop()

# Handle case where no documents were loaded
if rag_chain is None:
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
            
            # Save the conversation to a file
            save_conversation(prompt, response)

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

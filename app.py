
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pdf assisant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG ASSISTANT")
st.caption("Upload a PDF and ask questions about its content.")

# ---------------- MODELS ----------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )



@st.cache_resource
def get_llm():

    # First, check the local .env file
    groq_api_key = os.getenv("GROQ_API_KEY")

    # If not found, check Streamlit Cloud Secrets
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            groq_api_key = None

    # Check whether the API key exists
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please configure it in .env or Streamlit Secrets."
        )

    return ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-120b",
        temperature=0
    )



embedding_model = get_embedding_model()
llm = get_llm()

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.
        Use ONLY the provided context to answer the question.
        If the answer is not present in the context,
        say: "I could not find the answer in the document."
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])

# ---------------- DATABASE CREATION ----------------
def create_database(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()

    # A separate database for each PDF
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    db_path = str(Path("chroma_db_uploads") / pdf_hash)

    # Reuse the database if this PDF was already processed
    if os.path.exists(db_path):
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )
        return vectorstore

    # PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(
        suffix=".pdf", delete=False
    ) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No readable pages were found in the PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text could be extracted from the PDF.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=db_path
        )

        return vectorstore

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- SIDEBAR: PDF UPLOAD ----------------
with st.sidebar:
    st.header("📄 Upload your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

        if st.button("Process PDF", type="primary"):
            try:
                with st.spinner(
                    "Loading PDF, creating chunks and embeddings..."
                ):
                    vectorstore = create_database(uploaded_file)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = (
                        vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={
                                "k": 4,
                                "fetch_k": 10,
                                "lambda_mult": 0.5
                            }
                        )
                    )

                    st.session_state.processed_pdf = (
                        uploaded_file.name
                    )
                    st.session_state.processed_hash = (
                        hashlib.sha256(
                            uploaded_file.getvalue()
                        ).hexdigest()
                    )

                    # Start a fresh chat for a newly processed PDF
                    st.session_state.messages = []

                st.success("PDF processed successfully!")

            except Exception as e:
                st.error(f"Could not process PDF: {e}")

    if st.session_state.get("processed_pdf"):
        st.divider()
        st.success(
            f"Active document: {st.session_state.processed_pdf}"
        )

# ---------------- CHAT INTERFACE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

# Prevent chatting with an older PDF after selecting a new one
current_hash = (
    hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if uploaded_file is not None
    else None
)

if (
    st.session_state.get("retriever") is not None
    and current_hash is not None
    and current_hash != st.session_state.processed_hash
):
    st.info("Click **Process PDF** to activate the newly selected document.")

elif st.session_state.get("retriever") is None:
    st.info(
        "👈 Upload a PDF in the sidebar and click **Process PDF** "
        "to start chatting."
    )

else:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about your PDF...")

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching your document..."):
                try:
                    retriever = st.session_state.retriever

                    # Retrieve relevant chunks
                    docs = retriever.invoke(query)

                    # Combine retrieved text
                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                     # Temporary debugging
                    st.write("Retrieved chunks:", len(docs))

                    with st.expander("DEBUG: Retrieved context"):
                       st.write(context)

                    # Generate final prompt
                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": query
                    })

                    # Generate answer using Groq
                    response = llm.invoke(final_prompt)
                    answer = response.content

                    st.markdown(answer)
        

                    
                    # Optional: show source page numbers
                    with st.expander("View retrieved sources"):
                        for doc in docs:
                            page = doc.metadata.get("page")
                            if page is not None:
                                st.caption(f"Page {page + 1}")
                            st.write(doc.page_content[:500])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

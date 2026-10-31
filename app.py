
import os
import hashlib
import tempfile
import shutil
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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG ASSISTANT")
st.caption("Upload a PDF and ask questions about its content.")


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_pdf" not in st.session_state:
    st.session_state.processed_pdf = None

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None


# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


@st.cache_resource
def get_llm():

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            groq_api_key = None

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Configure it in Streamlit Secrets."
        )

    return ChatGroq(
        api_key=groq_api_key,
        model="openai/gpt-oss-120b",
        temperature=0
    )


embedding_model = get_embedding_model()
llm = get_llm()


# ============================================================
# PROMPT TEMPLATE
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer using ONLY the provided document context.

        If the answer is not present in the context,
        say:
        "I could not find the answer in the document."

        Do not invent information.
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


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    # Generate a unique ID for the uploaded PDF
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Use a writable temporary directory
    db_path = (
        Path(tempfile.gettempdir())
        / "chroma_db_uploads"
        / pdf_hash
    )

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save PDF temporarily for PyPDFLoader
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as temp_file:

        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:

        # ----------------------------------------------------
        # 1. LOAD PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        st.write("PDF pages extracted:", len(docs))

        if not docs:
            raise ValueError(
                "No readable pages found in this PDF."
            )

        extracted_characters = sum(
            len(doc.page_content.strip())
            for doc in docs
        )

        st.write(
            "Extracted characters:",
            extracted_characters
        )

        if extracted_characters == 0:
            raise ValueError(
                "No readable text found in the PDF. "
                "It may contain scanned images."
            )

        # ----------------------------------------------------
        # 2. SPLIT INTO CHUNKS
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        st.write("Chunks created:", len(chunks))

        if not chunks:
            raise ValueError(
                "No chunks were created from the PDF."
            )

        # ----------------------------------------------------
        # 3. CREATE FRESH CHROMA DATABASE
        # ----------------------------------------------------

        # Remove an old database for this PDF, if present
        shutil.rmtree(
            db_path,
            ignore_errors=True
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(db_path)
        )

        # ----------------------------------------------------
        # 4. VERIFY DATABASE
        # ----------------------------------------------------

        stored_count = vectorstore._collection.count()

        st.write(
            "ChromaDB stored documents:",
            stored_count
        )

        if stored_count == 0:
            raise ValueError(
                "ChromaDB contains zero documents."
            )

        return vectorstore

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# SIDEBAR: PDF UPLOAD
# ============================================================

with st.sidebar:

    st.header("📄 Upload your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.caption(
            f"Selected: {uploaded_file.name}"
        )

        if st.button(
            "Process PDF",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Loading PDF, creating chunks and embeddings..."
                ):

                    vectorstore = create_database(
                        uploaded_file
                    )

                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )

                    # Save active database and retriever
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever

                    st.session_state.processed_pdf = (
                        uploaded_file.name
                    )

                    st.session_state.processed_hash = (
                        hashlib.sha256(
                            uploaded_file.getvalue()
                        ).hexdigest()
                    )

                    # Reset previous conversation
                    st.session_state.messages = []

                st.success(
                    "PDF processed successfully!"
                )

            except Exception as e:

                st.session_state.retriever = None
                st.session_state.vectorstore = None
                st.session_state.processed_pdf = None
                st.session_state.processed_hash = None

                st.error(
                    f"Could not process PDF: {e}"
                )

    # Active document status
    if st.session_state.processed_pdf:

        st.divider()

        st.success(
            f"Active document: "
            f"{st.session_state.processed_pdf}"
        )


# ============================================================
# DOCUMENT VALIDATION
# ============================================================

current_hash = (
    hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()
    if uploaded_file is not None
    else None
)

if (
    st.session_state.retriever is not None
    and current_hash is not None
    and current_hash != st.session_state.processed_hash
):

    st.info(
        "Click **Process PDF** to activate the newly selected document."
    )

elif st.session_state.retriever is None:

    st.info(
        "👈 Upload a PDF and click **Process PDF** "
        "to start chatting."
    )


# ============================================================
# CHAT INTERFACE
# ============================================================

else:

    # Display conversation history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input(
        "Ask a question about your PDF..."
    )

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your document..."
            ):

                try:

                    # ------------------------------------------------
                    # 1. RETRIEVE DOCUMENTS
                    # ------------------------------------------------

                    retriever = st.session_state.retriever

                    docs = retriever.invoke(query)

                    st.write(
                        "Retrieved chunks:",
                        len(docs)
                    )

                    # ------------------------------------------------
                    # 2. BUILD CONTEXT
                    # ------------------------------------------------

                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                    )

                    # Debug information
                    with st.expander(
                        "DEBUG: Retrieved context"
                    ):

                        if context.strip():
                            st.write(context)
                        else:
                            st.warning(
                                "No context was retrieved."
                            )

                    # ------------------------------------------------
                    # 3. GENERATE ANSWER
                    # ------------------------------------------------

                    if not docs or not context.strip():

                        answer = (
                            "No relevant document chunks were retrieved. "
                            "Please reprocess the PDF or try another question."
                        )

                        st.warning(answer)

                    else:

                        final_prompt = prompt.invoke({
                            "context": context,
                            "question": query
                        })

                        response = llm.invoke(
                            final_prompt
                        )

                        answer = response.content

                        st.markdown(answer)

                    # ------------------------------------------------
                    # 4. DISPLAY SOURCES
                    # ------------------------------------------------

                    with st.expander(
                        "View retrieved sources"
                    ):

                        if docs:

                            for doc in docs:

                                page = doc.metadata.get("page")

                                if page is not None:

                                    st.caption(
                                        f"Page {page + 1}"
                                    )

                                st.write(
                                    doc.page_content[:500]
                                )

                        else:

                            st.write(
                                "No sources retrieved."
                            )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(
                        f"Error generating answer: {e}"
                    )

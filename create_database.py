# load pdf
# split into chunks
# create the embeddings
# stores in chroma
# phase 1
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document_loaders/deeplearning.pdf")
docs  = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)

chunks = splitter.split_documents(docs)

embedding_model  = HuggingFaceEmbeddings(
     model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(

    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma_db"
)
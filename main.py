from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory ="chroma_db",
    embedding_function = embedding_model
)

retriever  = vectorstore.as_retriever(
     search_type = "mmr",
     search_kwargs = {
         "k" : 4,
         "fetch_k" :10,
         "lambda_mult" : 0.5
     }
)

llm= ChatGroq(
    model="openai/gpt-oss-120b", temperature=0
)
# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""
           you are a helpful AI assistant
           use ONLY the provided context to answer the question.
           if the answer is not present in the context,
           say : "i could not find the answer in the document."
"""),
   ("human",
    """ context:
    {context}

    question:
    {question}
             """
     )
    ]
)
print(" RAG system is created!")
print("press 0 to exit")

while True:
    query  = input("Ask your documents query!: ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke(
        {
            "context" : context,
            "question" :query
        }
    )
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
     
         

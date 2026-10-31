# PDF RAG Chatbot

A PDF-based Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Hugging Face embeddings**, **ChromaDB**, **Groq**, and **Streamlit**. Upload a PDF, process it into searchable chunks, and ask questions about its content through a simple web interface.

## Features

- Upload a PDF through the Streamlit sidebar.
- Extract PDF text with `PyPDFLoader`.
- Split text into chunks using `RecursiveCharacterTextSplitter`.
- Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Store and retrieve document chunks using ChromaDB.
- Use Maximum Marginal Relevance (MMR) retrieval to find relevant context.
- Generate answers with Groq's `openai/gpt-oss-120b` model.
- Display retrieved source snippets and page numbers.
- Keep each uploaded PDF's vector database separate.

## How It Works

1. **Upload:** Choose a PDF in the app.
2. **Process:** The app loads the PDF, splits it into chunks, creates embeddings, and stores them in ChromaDB.
3. **Retrieve:** When you ask a question, the retriever selects relevant chunks using MMR.
4. **Generate:** The Groq model answers using the retrieved context.
5. **Display:** The answer and retrieved source snippets appear in the chat interface.

## Project Structure

```text
your-project/
├── app.py                 # Streamlit user interface and RAG workflow
├── main.py                # Original terminal-based RAG chatbot
├── create_database.py     # Original script to build the database
├── requirements.txt       # Python dependencies
├── .env                   # API key (keep private; do not commit)
├── .gitignore             # Files Git should ignore
├── document_loaders/
│   └── deeplearning.pdf   # Example PDF for create_database.py
├── chroma_db/              # Original Chroma database
└── chroma_db_uploads/      # Databases created for uploaded PDFs
```

Your actual files may differ. The Streamlit app uses `app.py`; the original scripts can remain in the project.

## Requirements

- Python version supported by your installed LangChain and Streamlit packages
- A Groq API key
- Internet access for the initial download of the Hugging Face embedding model and for Groq API calls
- `uv` recommended for dependency and virtual-environment management

## Setup with `uv`

### 1. Open the project directory

In the VS Code terminal, navigate to the folder containing your project files.

### 2. Create a `uv` project if needed

If the folder is not already a `uv` project:

```bash
uv init
```

If `pyproject.toml` already exists, do not run `uv init` again.

### 3. Install dependencies

If you have the `requirements.txt` included in this repository:

```bash
uv add -r requirements.txt
```

Alternatively, add the dependencies directly:

```bash
uv add streamlit python-dotenv pypdf langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface langchain-chroma langchain-groq sentence-transformers chromadb
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_actual_groq_api_key
```

Replace the placeholder with your own key. Do not share the key or commit `.env` to Git.

### 5. Run the Streamlit app

```bash
uv run streamlit run app.py
```

Streamlit normally opens the app at:

```text
http://localhost:8501
```

## Using the App

1. Open the app in your browser.
2. In the sidebar, choose a PDF file.
3. Click **Process PDF** and wait for processing to finish.
4. Enter a question in the chat input.
5. Review the answer and optionally expand **View retrieved sources**.

If you select a different PDF, click **Process PDF** before asking questions about it.

## RAG Configuration

The current configuration is:

| Setting | Value |
|---|---|
| Chunk size | `1000` characters |
| Chunk overlap | `200` characters |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Retrieval strategy | MMR |
| `k` | `4` |
| `fetch_k` | `10` |
| `lambda_mult` | `0.5` |
| Chat model | `openai/gpt-oss-120b` via Groq |
| Temperature | `0` |

The prompt instructs the model to use only the retrieved context and respond with **“I could not find the answer in the document.”** when the answer is not present in that context. As with any LLM application, this instruction helps guide behavior but does not guarantee that every answer will be perfectly grounded.

## Notes and Troubleshooting

- **API key errors:** Check that `.env` is in the project root and contains a valid `GROQ_API_KEY`.
- **Missing packages:** Run the dependency installation command from the project directory.
- **Slow first run:** The embedding model may need to download the first time.
- **PDF extraction issues:** Text-based PDFs work best. Scanned or image-only PDFs may need OCR before their text can be retrieved.
- **No relevant answer:** Try a more specific question and ensure the PDF was processed successfully.
- **Chroma persistence:** Keep the Chroma directories if you want to reuse the stored vectors. Avoid deleting them while the app is using them.
- **Large PDFs:** Processing can take time and use additional memory, especially when embeddings are created.

## Security

- Never commit `.env`, API keys, or other secrets.
- Do not upload confidential PDFs to a deployment unless you have permission and understand the hosting provider's privacy and retention policies.
- If deploying publicly, consider file-size limits, PDF validation, rate limiting, and user/session isolation.

## Git Ignore

A basic `.gitignore` can include:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
chroma_db/
chroma_db_uploads/
.streamlit/secrets.toml
```

Only ignore the Chroma directories if you do not intend to version or share the generated vector databases.

## License

Add a license file if you plan to publish or distribute this project. Until you choose a license, reuse and redistribution permissions may be limited.

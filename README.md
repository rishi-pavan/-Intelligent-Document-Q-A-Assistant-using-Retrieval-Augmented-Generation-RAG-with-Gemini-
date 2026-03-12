🧠 RAG Assistant: HF Embeddings + Gemini LLM + FAISS
Retrieval-Augmented Generation (RAG) Assistant for Document Question Answering

🚀 Project Overview This project implements a RAG (Retrieval-Augmented Generation) pipeline using:

📄 PDF Document Upload

🧩 Text Chunking

🧠 Semantic Search with Hugging Face Embeddings (all-MiniLM-L6-v2)
⚡ Vector Storage using FAISS (local, in-memory vector database)
✍️ Answer Generation using Google Gemini (gemini-1.5-pro-latest)
🎯 Interactive Streamlit UI for user interaction
The assistant reads the uploaded document, breaks it into chunks, generates embeddings, stores them in FAISS, retrieves the most relevant chunks for any user query, and uses Gemini to generate a concise, context-based answer.

🛠️ Tech Stack
Component & Technology Involved:

Vector Database: FAISS (in-memory)
Embedding Model: Hugging Face (all-MiniLM-L6-v2)
Language Model: Google Gemini (gemini-1.5-pro-latest)
Document Loader: PyPDF2 for PDF reading
UI: Streamlit
Chunking: LangChain’s CharacterTextSplitter
📂 How It Works
User uploads a PDF document through the Streamlit interface.
The document is split into smaller chunks for efficient processing.
Each chunk is embedded into vectors using Hugging Face Embeddings.
FAISS stores these embeddings and provides semantic retrieval for user queries.
When the user asks a question:
The system retrieves top matching document chunks.
These chunks are passed as context to Gemini LLM.
Gemini generates the final answer based on the retrieved content.
✅ Getting Started

Clone the Repository git clone https://github.com/yourusername/rag-assistant.git cd rag-assistant

Set Up Virtual Environment

python -m venv .venv
source .venv/bin/activate #On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Create a .env File
GOOGLE_API_KEY=your_google_gemini_api_key
Make sure you have enabled Generative AI Studio and generated an API key for Gemini.

🟢 Run the App - streamlit run app.py

⚡ Example Usage
Upload a PDF document.
Ask questions like:
"Summarize the key points of this document."
"What does the agreement say about payment terms?"
Get precise, concise answers powered by RAG!
💡 Key Features

No need to fine-tune the LLM on your data.
Reduces hallucination by grounding the answers in your document.
Local and fast vector search using FAISS.
Supports any text-based PDFs (scanned PDFs with images are not supported unless OCR is added).
🔒 Limitations

Only handles PDF documents (not DOCX, CSV, etc.).
FAISS index is in-memory (optional persistence can be added).
No user authentication (for now).
📌 Future Improvements

Multi-file document support.
Persistent FAISS index storage.
Add OCR for scanned PDFs.
Support for other vector stores (Chroma, Pinecone).
Enhanced metadata-based filtering.
📬 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
🏆 Credits

LangChain[https://github.com/langchain-ai/langchain]
FAISS by Facebook AI Research[https://github.com/facebookresearch/faiss]
Google Gemini Generative AI[https://ai.google.dev/]
Sentence Transformers (Hugging Face)[https://www.sbert.net/]
⚖️ License This project is licensed under the MIT License.

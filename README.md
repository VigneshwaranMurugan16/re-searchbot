# 🔍 RE-SearchBot

> **An AI-powered Research Paper Analysis & Question-Answering Tool**

RE-SearchBot is an intelligent system that allows users to **upload research papers (PDF, DOCX, PPTX)** and ask **questions** about their content.  
It processes the document, splits it into text chunks, generates vector embeddings using **HuggingFace models**, and stores them in a **FAISS vector database**.  
The backend is built with **FastAPI**, while the frontend (React.js) handles file uploads and interaction.  

---

## 🚀 Key Features

- 📄 Upload research documents (`.pdf`, `.docx`, `.pptx`)
- ⚙️ Custom text loaders for DOCX and PPTX formats  
- 🧠 Embedding generation with `HuggingFaceBgeEmbeddings`  
- 🗂️ Document chunking with `RecursiveCharacterTextSplitter`  
- 💾 FAISS-based vector database for fast semantic search  
- 🌐 REST API with FastAPI  
- 🧩 Streamlit interface for question answering  
- 🔐 CORS enabled for frontend-backend communication  

---

## 🏗️ Project Structure (Backend Focus)
```
RE-SearchBot/
│
├── backend/
│ ├── main.py # FastAPI backend logic
│ ├── custom_loaders.py # Custom loaders for DOCX and PPTX files
│ ├── uploaded_files/ # Temporary folder for uploads
│ ├── Main_Files/ # Stores FAISS index and pickle files
│ ├── requirements.txt # Python dependencies
│ └── ...
│
└── frontend/
├── src/
├── package.json
└── ...
```

---

## ⚙️ Backend Setup (FastAPI)
```bash
### 1️⃣ Create & Activate Virtual Environment
bash
cd backend
python -m venv venv
# Activate
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run FastAPI Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

FastAPI will be running at:
👉 http://127.0.0.1:8000


## ⚙️ Frontend Setup (React.js)
cd frontend
npm install
npm start

Runs the React frontend at 👉 http://localhost:3000

## 🧠 Backend Workflow
--Upload & Cleanup
--User uploads a document through the frontend (or directly via API).
--Backend saves it to uploaded_files/ and clears any previous uploads.
--Document Processing
--Depending on file type (pdf, docx, pptx), the correct loader is used:
--PyPDFLoader for PDFs
--DocxLoader for DOCX
- PptxLoader for PPTX
- Text Chunking
- Uses RecursiveCharacterTextSplitter to break text into smaller overlapping segments for context preservation.
- Embedding Generation
- Embeddings are generated using HuggingFaceBgeEmbeddings (BAAI/bge-small-en-v1.5) on CPU.
- Vector Database Creation
- FAISS creates a local vector index and saves it as index.pkl and index.faiss in Main_Files/.
- Launching Streamlit Q&A
- The /start endpoint spawns a Streamlit process, providing an interactive interface to query the indexed data.

## 🧾 Example Usage
Step 1 — Run FastAPI Backend
uvicorn main:app --reload
Step 2 — Run React Frontend
npm start

Step 3 — Upload Research Paper
- Open frontend (http://localhost:3000)
- Upload a .pdf, .docx, or .pptx research paper
- Backend will create FAISS embeddings automatically

Step 4 — Start Q&A
curl -X POST http://127.0.0.1:8000/start
Then open Streamlit at http://localhost:8501 to interactively ask questions.

## 🧩 Dependencies

Main dependencies for backend:
- fastapi
- uvicorn
- langchain
- langchain_community
- faiss-cpu
- huggingface_hub
- pypdf
- python-docx
- python-pptx
- streamlit



## Output
<img width="1268" height="657" alt="researchbotss1" src="https://github.com/user-attachments/assets/1b53998e-59e4-4aaa-b3c6-aa2c2acea53e" />
<img width="1268" height="659" alt="researchbotss2" src="https://github.com/user-attachments/assets/b930445b-75e5-41c5-8faa-ac406b544d3e" />



##💡 Acknowledgements

- FastAPI
- LangChain
- FAISS
- HuggingFace Transformers
- Streamlit


## 📝 License
This project is created by Vigneshwaran Murugan for educational and research purposes.
This project is licensed under the GNU General Public License v3.0 (GPLv3)  
See the [LICENSE](LICENSE) file for details.



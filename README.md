# UNICHUNK - A PDF Insight Assistant

A sophisticated Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides intelligent, context-aware responses with visual highlighting, table extraction, and multi-modal analysis capabilities.

## 🌟 Features

### Core Functionality
- **Multi-Modal Document Processing**: Handles text, tables, and visual content from PDFs
- **Intelligent Query Classification**: Automatically routes queries based on intent and content type
- **Visual Highlighting**: Highlights relevant document sections with coordinate-based precision
- **Table Extraction & Analysis**: Extracts and processes tabular data into DataFrames
- **Plot Generation**: Creates visualizations based on document data
- **Conversation History**: Maintains session-based Q&A history
- **Vision-Language Model Integration**: Analyzes images and diagrams using VLMs

### Technical Architecture
- **LangGraph Orchestration**: State-managed pipeline execution
- **FAISS Vector Store**: Efficient similarity search and retrieval
- **Streamlit Frontend**: Interactive web interface
- **Ollama Integration**: Local LLM inference
- **Sentence Transformers**: State-of-the-art embeddings

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (Download from [python.org](https://www.python.org/downloads/))
- Git for Windows (Download from [git-scm.com](https://git-scm.com/download/win))
- Ollama for Windows (Download from [ollama.ai](https://ollama.ai))
- 8GB+ RAM recommended
- PowerShell or Command Prompt

### Installation

1. **Clone the repository**
```powershell
git clone <repository-url>
cd pdf-insight-assistant
```

2. **Install Poetry** (if not already installed)
```powershell
# Using PowerShell (Recommended)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Alternative: Using pip
pip install poetry

# Add Poetry to PATH if needed
# Add C:\Users\<username>\AppData\Roaming\Python\Scripts to your PATH
```

3. **Install dependencies with Poetry**
```powershell
poetry install
```

4. **Activate the Poetry environment**
```powershell
poetry shell
```

5. **Install and setup Ollama models**
```powershell
# Download and install Ollama from https://ollama.ai
# After installation, run these commands in Command Prompt or PowerShell:
ollama pull phi3:3.8b
ollama pull qwen2.5vl:7b
```

### Dependencies (pyproject.toml)

```toml
[tool.poetry.dependencies]
python = "^3.8"
streamlit = "^1.28.0"
langgraph = "^0.0.40"
sentence-transformers = "^2.2.2"
faiss-cpu = "^1.7.4"
unstructured = {extras = ["pdf"], version = "^0.10.0"}
ollama = "^0.1.7"
matplotlib = "^3.7.0"
pandas = "^2.0.0"
Pillow = "^10.0.0"
PyMuPDF = "^1.23.0"
pydantic = "^2.0.0"
numpy = "^1.24.0"
tqdm = "^4.65.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.0.0"
flake8 = "^6.0.0"
mypy = "^1.5.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

## 🏃‍♂️ Running the Application

### Start the Streamlit Interface
```powershell
# Make sure you're in the Poetry environment
poetry shell

# Run the application
poetry run streamlit run app.py
```

### Alternative: Direct Poetry Run
```powershell
poetry run streamlit run app.py
```

### Access the Application
Open your browser and navigate to `http://localhost:8501`

### Development Commands
```powershell
# Run tests
poetry run pytest

# Code formatting
poetry run black .

# Type checking
poetry run mypy .

# Linting
poetry run flake8 .
```

## 📖 Usage Guide

### 1. Upload PDF Document
- Click "Upload a PDF" in the interface
- Select your PDF file (supports multi-page documents)
- Wait for the ingestion pipeline to complete

### 2. Ask Questions
- Type your question in the input field
- Questions can be about:
  - General content ("What is this document about?")
  - Specific data ("Show me the financial tables")
  - Visual content ("Describe the diagrams")
  - Analysis requests ("Generate a chart of the revenue data")

### 3. View Results
- **Text Answers**: Contextual responses based on document content
- **Highlighted Pages**: Visual highlighting of relevant sections
- **Data Tables**: Extracted tabular data as DataFrames
- **Generated Plots**: Visualizations created from document data

## 🏗️ Architecture Overview

### Ingestion Pipeline
```
PDF Upload → Document Loader → Content Classifier → Parallel Processing
                                                   ├── Text Chunker
                                                   ├── Table Chunker  
                                                   └── Visual Chunker
                                        ↓
                            Chunk Aggregator → Embedding Generation → Vector Store
```

### Query Pipeline
```
User Query → Query Classifier → Modality Classifier → Query Embedder
                                                    ↓
          Answer Generator ← Highlight Generator ← Retriever
                    ↓
              Enrichment (Tables/Plots) → Final Response
```

## 🔧 Configuration

### Model Configuration
- **Text LLM**: `phi3:3.8b` (configurable in nodes)
- **Vision LLM**: `qwen2.5vl:7b` (configurable in `visual_chunker.py`)
- **Embedding Model**: `all-MiniLM-L6-v2` (configurable in state)

### Processing Parameters
- **Chunk Size**: 300 words (configurable in `text_chunker.py`)
- **Retrieval Count**: Top 10 chunks (configurable in `retriever.py`)
- **Image DPI**: 200 (configurable in `visual_chunker.py`)

## 📁 Project Structure

```
pdf-insight-assistant\
├── app.py                          # Streamlit frontend
├── state.py                        # Pydantic state models
├── retrieval_graph_config.py       # Query pipeline configuration
├── ingestion_graph_config.py       # Ingestion pipeline configuration
├── streamlit_backend_connector.py  # Pipeline orchestration
├── tools.py                        # Utility functions
├── nodes\
│   ├── ingestion_nodes\
│   │   ├── document_loader.py      # PDF parsing with unstructured
│   │   ├── content_classifier.py   # Content type detection
│   │   ├── text_chunker.py         # Text processing and chunking
│   │   ├── table_chunker.py        # Table extraction
│   │   ├── visual_chunker.py       # Image analysis with VLM
│   │   ├── chunk_aggregator.py     # Chunk consolidation
│   │   └── embedding_node.py       # Vector embedding generation
│   └── retrieval_nodes\
│       ├── index_loader.py         # Vector store loading
│       ├── query_classifier.py     # Intent classification
│       ├── query_modality_classifier.py # Content type routing
│       ├── query_embedder.py       # Query vectorization
│       ├── retriever.py           # Similarity search
│       ├── highlight_node.py       # Coordinate extraction
│       ├── answer_generator.py     # Response generation
│       ├── pandas_node.py          # Table processing
│       └── plotting_node.py        # Visualization generation
├── vector_store\                   # Generated embeddings and metadata
├── highlighted_pages\              # Generated highlight images
├── pyproject.toml                  # Poetry configuration
└── poetry.lock                     # Poetry lock file
```

## 🧪 Testing

### Manual Testing
1. Upload test PDFs with various content types
2. Test different query types:
   - General questions
   - Table extraction requests
   - Visual content queries
   - Plot generation requests

### Expected Inputs/Outputs

**Input**: PDF documents containing text, tables, and/or images
**Output**: 
- Contextual text responses
- Highlighted document pages
- Extracted data tables
- Generated visualizations
- Conversation history

## 🔍 Troubleshooting

### Common Issues

1. **Ollama Models Not Found**
   ```powershell
   ollama pull phi3:3.8b
   ollama pull qwen2.5vl:7b
   ```

2. **Poetry Environment Issues**
   ```powershell
   # Reset Poetry environment
   poetry env remove python
   poetry install
   poetry shell
   ```

3. **Poetry Not Found After Installation**
   ```powershell
   # Add Poetry to PATH manually
   # Add this path to your system PATH:
   # C:\Users\<your-username>\AppData\Roaming\Python\Scripts
   
   # Or restart PowerShell/Command Prompt
   # Or reinstall using pip:
   pip install poetry
   ```

4. **Python Not Found**
   ```powershell
   # Make sure Python is installed and in PATH
   python --version
   
   # If not found, add Python to PATH or reinstall from python.org
   ```

5. **Memory Issues with Large PDFs**
   - Reduce chunk size in `text_chunker.py`
   - Process fewer pages at once

6. **Coordinate Highlighting Errors**
   - Check PDF structure with `tools.py` utilities
   - Verify coordinate serialization

7. **Slow Processing**
   - Use GPU-enabled FAISS if available
   - Reduce image DPI in visual processing

8. **Module Import Errors**
   ```powershell
   # Ensure you're in Poetry environment
   poetry shell
   # Or run directly with Poetry
   poetry run python -c "import streamlit; print('OK')"
   ```

9. **PowerShell Execution Policy Issues**
   ```powershell
   # If you get execution policy errors, run:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

10. **Port Already in Use**
    ```powershell
    # If port 8501 is busy, specify a different port:
    poetry run streamlit run app.py --server.port 8502
    ```

### Performance Optimization
- Use GPU acceleration for embeddings
- Implement async processing for large documents
- Cache embeddings for repeated queries
- Optimize chunk sizes based on document type


## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [Unstructured](https://github.com/Unstructured-IO/unstructured) for document parsing
- Powered by [Ollama](https://ollama.ai) for local LLM inference
- Interface built with [Streamlit](https://streamlit.io)

---

**Note**: This system is designed for local deployment and testing. For production use, consider security implications, scalability requirements, and regulatory compliance based on your specific use case.


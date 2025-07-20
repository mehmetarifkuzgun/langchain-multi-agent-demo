# Multi-Agent System with LangChain, Ollama, and RAG

A sophisticated multi-agent system that combines LangChain, Ollama's Llama3.1:8b model, and Retrieval-Augmented Generation (RAG) for intelligent content creation and document analysis.

## 🚀 Features

### Core Agents
- **RAG Agent**: Document retrieval and knowledge-augmented responses using FAISS vector store
- **Research Agent**: Conducts comprehensive topic research with RAG enhancement
- **Writer Agent**: Creates structured content (articles, reports) based on research findings
- **Critic Agent**: Reviews and scores content with detailed feedback (1-10 scale)
- **Coordinator Agent**: Orchestrates multi-phase workflows with progress tracking

### Advanced Capabilities
- **Vector-based Document Retrieval**: FAISS integration for semantic search
- **JSON-structured Responses**: Standardized output format across all agents
- **Interactive Interfaces**: Both command-line and Streamlit web UI
- **Document Management**: Load from files/directories or add documents programmatically
- **Workflow Visualization**: Real-time progress tracking and results display

## 📋 Prerequisites

1. **Install Ollama**: Download from [https://ollama.ai/](https://ollama.ai/)
2. **Pull the model**:
   ```bash
   ollama pull llama3.1:8b
   ```
3. **Verify installation**:
   ```bash
   ollama serve
   ```

## 🔧 Installation

1. **Clone and navigate**:
   ```bash
   git clone <repository-url>
   cd langchain-demo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Option 1: Command Line Interface
```bash
# Run the complete demo with sample documents
python multi_agent_system.py

# Interactive mode with menu options
python interactive_demo.py
```

### Option 2: Streamlit Web Interface
```bash
streamlit run streamlit_ui.py
```
Then open your browser to `http://localhost:8501`

### Option 3: Programmatic Usage
```python
from multi_agent_system import MultiAgentSystem

# Initialize system
system = MultiAgentSystem()

# Add documents to RAG
documents = ["Your document content here..."]
system.add_documents_to_rag(documents)

# Run complete workflow
results = system.run_workflow("Your research topic")

# Run individual agents
research_result = system.run_single_agent("research", "AI in education")
```

## 🔄 Workflow Process

1. **RAG Retrieval**: Searches document store for relevant information
2. **Research Phase**: Generates comprehensive research summary with key concepts
3. **Writing Phase**: Creates structured content (title, intro, body, conclusion)
4. **Review Phase**: Provides detailed critique with numerical scoring
5. **Coordination**: Combines all phases with metadata and status tracking

## 📁 Project Structure

```
langchain-demo/
├── multi_agent_system.py    # Core agent implementations
├── streamlit_ui.py          # Web interface with visualizations
├── interactive_demo.py      # Command-line interface
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Configuration

Edit `config.py` to customize:
- **Model settings**: Temperature, max tokens, timeout
- **Workflow behavior**: Iterations, quality thresholds
- **Output options**: Save results, directory paths

## 📊 Output Format

All agents return structured JSON responses:
```json
{
  "agent_name": "Research Agent",
  "content": "{\"summary\": \"...\", \"key_concepts\": [...]}",
  "metadata": {"task_type": "research", "rag_enhanced": true},
  "timestamp": 1642742400.0
}
```

## 🎮 Example Use Cases

- **Academic Research**: Analyze documents and generate research summaries
- **Content Creation**: Research → Write → Review workflow for articles
- **Document Analysis**: RAG-powered Q&A on large document collections
- **Knowledge Management**: Intelligent document search and synthesis
- **Educational Tools**: Interactive learning with AI tutoring capabilities

## 🔍 RAG System Details

- **Embeddings**: Ollama embeddings with Llama3.1:8b
- **Vector Store**: FAISS for efficient similarity search
- **Text Splitting**: Recursive character splitting (1000 chars, 200 overlap)
- **Document Types**: Text files, directories, programmatic text input

## 🚨 Troubleshooting

**Common Issues:**
- Ensure Ollama service is running (`ollama serve`)
- Verify model is downloaded (`ollama list`)
- Check Python version compatibility (3.8+)
- Install all requirements (`pip install -r requirements.txt`)

**Performance Tips:**
- Adjust chunk size in RAG configuration for different document types
- Modify temperature settings for more/less creative outputs
- Use quality thresholds to control content approval

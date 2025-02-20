# 📄 AI PDF Chatbot: Intelligent Document Analysis Tool

## 🌟 Project Overview

This AI-powered PDF Chatbot is an advanced document analysis tool that leverages cutting-edge technologies to provide intelligent, context-aware interactions with PDF documents. Built using Retrieval-Augmented Generation (RAG), LangChain, and Perplexity AI, this application allows users to upload PDFs and engage in natural language conversations about their content.

## ✨ Key Features

- 📖 PDF Document Upload
- 🤖 Intelligent Summarization
- 💬 Context-Aware Question Answering
- 🧠 RAG (Retrieval-Augmented Generation) Implementation
- 🖥️ User-Friendly Streamlit Interface

## 🛠 Technologies Used

- **Language Model**: Perplexity AI
- **Framework**: LangChain
- **UI**: Streamlit
- **Embedding**: HuggingFace Embeddings
- **Vector Store**: FAISS
- **Programming Language**: Python

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aban-ali/ai-pdf-chatbot.git
cd ai-pdf-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r req.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY='your-perplexity-api-key'
```

5. 🖥️ Running the Application
``` bash
streamlit run app.py
```

## 📋 Usage

- Upload a PDF file
- Click "Process PDF"
- View the automatic summary
- Ask questions about the document

## 🔍 How It Works

- PDF Processing: Converts PDF to text chunks
- Embedding: Creates vector representations of document text
- Retrieval: Finds most relevant document sections
- Generation: Uses RAG to provide context-aware responses
 
## 🌈 Example Queries
- "What is the main topic of this document?"
- "Summarize the key points in section 2"
- "Explain the methodology described in this paper"

## 🐛 Known Issues
- Large PDFs might take longer to process
- Complex documents may require multiple queries for full understanding

## 📊 Performance Considerations
- Recommended PDF size: < 50 MB
- Best performance with well-structured documents
- Internet connection required for AI processing

## 📜 License
Distributed under the MIT License. See LICENSE for more information.

## 🌍 Connect & Support
- Created by Mohammad Aban Ali
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/mohammad-aban-ali-78592b2a7/)
- Project Link: https://github.com/aban-ali/ai-pdf-chatbot

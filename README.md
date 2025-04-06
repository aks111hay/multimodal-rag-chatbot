# Multimodal RAG Chatbot 🤖📄🔍

This project is a document-based, multimodal chatbot that uses Retrieval-Augmented Generation (RAG) to answer user questions from uploaded documents.

## 🚀 Features
- 🧠 Uses Hugging Face Transformers and FAISS for semantic search and answer generation.
- 📄 Supports `.txt` and `.pdf` documents for context.
- 💬 Interactive Gradio-based interface with both text and speech input/output.
- 🎙️ Integrates `gTTS` and `SpeechRecognition` for multimodal communication.

## 🗃️ How It Works
1. Upload a document (e.g. company policy).
2. Ask any question based on the document.
3. The chatbot retrieves relevant sections and answers intelligently.

## 🧪 Example Questions (Using Provided `TechNova_Policy.pdf`)
- "What is the vacation policy?"
- "What are the working hours at TechNova?"
- "Who should I contact for IT issues?"
- "What are TechNova’s company values?"
- "What happens when an employee resigns?"

## 📦 Installation

```bash
git clone https://github.com/aks111hay/multimodal-rag-chatbot.git
cd multimodal-rag-chatbot
pip install -r requirements.txt
python app.py

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
4. ![Screenshot 2025-04-06 134854](https://github.com/user-attachments/assets/77c752ae-e188-42f7-98df-3bdfc6814766)
5. ![Screenshot 2025-04-06 135009](https://github.com/user-attachments/assets/4fafb9ce-c6df-4f6e-b6da-78546fdfcddf)


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
```

## 📁 Project Structure 
├── app.py                  # Main chatbot script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── sample_docs/
    ├── sample_doc.txt
    └── TechNova_Policy.pdf
## Made with ❤️ by Akshay Kumar

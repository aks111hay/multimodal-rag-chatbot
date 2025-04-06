import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from gtts import gTTS
import uuid
import os
import PyPDF2

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Store index and mapping globally
index = None
doc_id_map = {}

def load_document(file):
    global index, doc_id_map
    
    # Extract text from file
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = file.read().decode("utf-8")

    # Split text into chunks
    chunks = text.split(". ")  # basic sentence splitting
    embeddings = embedder.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Create doc mapping
    doc_id_map = {i: chunk for i, chunk in enumerate(chunks)}

    return "Document processed. You can now ask questions."

def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return " ".join([doc_id_map[i] for i in I[0]])

def answer_question(query):
    if index is None:
        return "Please upload a document first.", None
    context = retrieve_context(query)
    result = qa_pipeline(question=query, context=context)
    answer = result["answer"]
    audio_file = synthesize_speech(answer)
    return answer, audio_file

def synthesize_speech(text):
    filename = f"response_{uuid.uuid4()}.mp3"
    tts = gTTS(text)
    tts.save(filename)
    return filename

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄🔍 Multimodal RAG Chatbot – Upload a Document and Ask Questions")

    with gr.Row():
        file_input = gr.File(label="Upload Document (.txt or .pdf)")
        doc_status = gr.Textbox(label="Status")

    file_input.change(fn=load_document, inputs=file_input, outputs=doc_status)

    query_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")
    audio_output = gr.Audio(type="filepath", label="Answer (Speech)")

    query_input.submit(fn=answer_question, inputs=query_input, outputs=[answer_output, audio_output])

demo.launch()
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from gtts import gTTS
import uuid
import os
import PyPDF2

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Store index and mapping globally
index = None
doc_id_map = {}

def load_document(file):
    global index, doc_id_map
    
    # Extract text from file
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = file.read().decode("utf-8")

    # Split text into chunks
    chunks = text.split(". ")  # basic sentence splitting
    embeddings = embedder.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # Create doc mapping
    doc_id_map = {i: chunk for i, chunk in enumerate(chunks)}

    return "Document processed. You can now ask questions."

def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    return " ".join([doc_id_map[i] for i in I[0]])

def answer_question(query):
    if index is None:
        return "Please upload a document first.", None
    context = retrieve_context(query)
    result = qa_pipeline(question=query, context=context)
    answer = result["answer"]
    audio_file = synthesize_speech(answer)
    return answer, audio_file

def synthesize_speech(text):
    filename = f"response_{uuid.uuid4()}.mp3"
    tts = gTTS(text)
    tts.save(filename)
    return filename

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄🔍 Multimodal RAG Chatbot – Upload a Document and Ask Questions")

    with gr.Row():
        file_input = gr.File(label="Upload Document (.txt or .pdf)")
        doc_status = gr.Textbox(label="Status")

    file_input.change(fn=load_document, inputs=file_input, outputs=doc_status)

    query_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer")
    audio_output = gr.Audio(type="filepath", label="Answer (Speech)")

    query_input.submit(fn=answer_question, inputs=query_input, outputs=[answer_output, audio_output])

demo.launch()

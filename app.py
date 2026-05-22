
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local('vectorstore')
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    db = FAISS.load_local(
        'vectorstore',
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

def get_llm():
    generator = pipeline(
        'text2text-generation',
        model='google/flan-t5-base',
        max_length=256
    )
    return generator
def answer_question(question, db, llm):

    docs = db.similarity_search(question, k=3)

    context = '\n'.join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm(prompt)

    return response[0]['generated_text']

st.set_page_config(page_title='PDF Chatbot')
st.title('📄 PDF Question Answer Bot')

uploaded_file = st.file_uploader(
    'Upload a PDF file',
    type='pdf'
)

if uploaded_file:
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    with open(f'/content/HR-Manual-Full-Book.pdf}', 'wb') as f:
        f.write(uploaded_file.read())

    st.success('PDF uploaded successfully!')

    if st.button('Process PDF'):
        with st.spinner('Processing PDF...'):
            documents = load_pdf(f'/content/HR-Manual-Full-Book.pdf')
            chunks = split_text(documents)
            create_vector_store(chunks)
        st.success('PDF processed successfully!')

question = st.text_input('Ask a question from the PDF')

if question:
    db = load_vector_store()
    llm = get_llm()
    answer = answer_question(question, db, llm)
    st.write('### Answer')
    st.write(answer)

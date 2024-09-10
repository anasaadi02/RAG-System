import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
import os

# Configuration de la page
st.set_page_config(page_title="Application PDF Upload", layout="centered")

# Embedding function
emb_function = OllamaEmbeddings(model="all-minilm")

# PDF Upload Page
def upload_page():
    st.header("Discuter avec un PDF")
    
    pdf = st.file_uploader("Téléchargez votre PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Split the text into chunks
        text_split = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_split.split_text(text=text)
        
        storeName = pdf.name[:-4]
        
        # Save or load embeddings
        if os.path.exists(f"{storeName}.pkl"):
            with open(f"{storeName}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
            st.write("Embeddings chargés depuis le disque")
        else:
            vectorStore = FAISS.from_texts(chunks, embedding=emb_function)
            with open(f"{storeName}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)
            st.write("Embeddings complétés")
        
        query = st.text_input("Posez vos questions ici : ")
        
        if query:
            docs = vectorStore.similarity_search(query=query, k=3)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            st.write("Contexte trouvé :")
            st.write(context)

if __name__ == "__main__":
    upload_page()

import os
import hashlib
import PyPDF2
import streamlit as st
from streamlit.logger import get_logger

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pinecone import Pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


LOGGER = get_logger(__name__)

# Set your API keys and environment variables
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME = st.secrets['PINECONE_INDEX_NAME']

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Google Drive Authentication and Initialization
def initialize_google_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
    drive = GoogleDrive(gauth)
    return drive

drive = initialize_google_drive()

def pdf_to_text(file_path: str) -> str:
    """Convert a PDF file from Google Drive to text."""
    with open(file_path, "rb") as f:
        pdfReader = PyPDF2.PdfReader(f)
        count = len(pdfReader.pages)
        text = ""
        for i in range(count):
            page = pdfReader.pages[i]
            text += page.extract_text()
    return text

def embed(text: str, filename: str):
    """Generate embeddings from text and store them in Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len, 
        is_separator_regex=False
    )
    docs = text_splitter.create_documents([text])
    for idx, d in enumerate(docs):
        hash = hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        embedding = client.embeddings.create(
            model="text-embedding-ada-002", 
            input=d.page_content
        ).data[0].embedding
        metadata = {
            "hash": hash,
            "text": d.page_content,
            "index": idx,
            "model": "text-embedding-ada-003",
            "docname": filename
        }
        index.upsert([(hash, embedding, metadata)])

def process_and_upload_from_drive(file_ids: list):
    """Process PDF files from Google Drive and upload embeddings to Pinecone."""
    for file_id in file_ids:
        # Download the file from Google Drive
        file = drive.CreateFile({'id': file_id})
        file.GetContentFile(file['title'])
        text = pdf_to_text(file['title'])

        # Generate and store embeddings in Pinecone
        embed(text, file['title'])

# Example usage: Process specific Google Drive files
file_ids = ["your-google-drive-file-id-1", "your-google-drive-file-id-2"]
process_and_upload_from_drive(file_ids)

st.markdown("# Documents processed and embeddings uploaded to Pinecone.")

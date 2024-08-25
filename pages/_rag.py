import os
import hashlib
import streamlit as st
from streamlit.logger import get_logger

from langchain.document_loaders import GoogleDriveLoader
from langchain.retrievers import GoogleDriveRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from openai import OpenAI

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

# Initialize GoogleDriveLoader and GoogleDriveRetriever
loader = GoogleDriveLoader(
    folder_id="1J1wKGZJQIKen33xaVEBI8FpK8B4WQzIJ",  # Replace with your folder ID
    credentials="pages/credentials.json"  # Replace with your path to credentials
)
retriever = GoogleDriveRetriever(loader=loader)

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

def process_and_upload_from_google_drive():
    """Process files from Google Drive and upload embeddings to Pinecone."""
    documents = retriever.retrieve(query="")
    for doc in documents:
        LOGGER.info(f"Processing file: {doc.metadata['title']}")
        embed(doc.page_content, doc.metadata['title'])

# Process all files in the specified Google Drive folder
process_and_upload_from_google_drive()

st.markdown("# Documents in the folder processed and embeddings uploaded to Pinecone.")

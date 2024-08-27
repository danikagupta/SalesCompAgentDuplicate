import os
import hashlib
import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.document_loaders import GoogleDriveLoader
from pinecone import Pinecone
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Set your API keys and environment variables
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_API_ENV = st.secrets['PINECONE_API_ENV']
PINECONE_INDEX_NAME = st.secrets['PINECONE_INDEX_NAME']

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Define the scope for Google Drive API access
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_google_drive():
    creds = None
    # Check if token.json exists to reuse credentials
    if os.path.exists('.credentials/token.json'):
        creds = Credentials.from_authorized_user_file('.credentials/token.json', SCOPES)
    if not creds or not creds.valid:
        # If no valid credentials, initiate the OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(
            '.credentials/credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('.credentials/token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

# Authenticate and create the service object
creds = authenticate_google_drive()
service = build('drive', 'v3', credentials=creds)

# Initialize GoogleDriveLoader with the service object
loader = GoogleDriveLoader(
    folder_id="1J1wKGZJQIKen33xaVEBI8FpK8B4WQzIJ",  # Replace with your Google Drive folder ID
    service=service
)

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
    documents = loader.load()  # Load documents from Google Drive
    for doc in documents:
        st.write(f"Processing file: {doc.metadata['title']}")
        embed(doc.page_content, doc.metadata['title'])

# Process all files in the specified Google Drive folder
process_and_upload_from_google_drive()

st.markdown("# Documents in the folder processed and embeddings uploaded to Pinecone.")



"""
import os
import hashlib
import streamlit as st
from streamlit.logger import get_logger

from langchain.document_loaders import GoogleDriveLoader
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

# Initialize GoogleDriveLoader
loader = GoogleDriveLoader(
    folder_id="1J1wKGZJQIKen33xaVEBI8FpK8B4WQzIJ",  # Replace with your folder ID
    credentials=".credentials/credentials.json"  # Path to your Google OAuth credentials
)

def embed(text: str, filename: str):
    ""Generate embeddings from text and store them in Pinecone.""
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
    ""Process files from Google Drive and upload embeddings to Pinecone.""
    documents = loader.load()  # Load documents from Google Drive
    for doc in documents:
        LOGGER.info(f"Processing file: {doc.metadata['title']}")
        embed(doc.page_content, doc.metadata['title'])

# Process all files in the specified Google Drive folder
process_and_upload_from_google_drive()

st.markdown("# Documents in the folder processed and embeddings uploaded to Pinecone.")
"""
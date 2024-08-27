from langchain.document_loaders import GoogleDriveLoader
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

import os
import io

# Set up the OAuth 2.0 flow
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_gdrive():
    creds = None
    flow = InstalledAppFlow.from_client_secrets_file(
        '.credentials/credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

def main():
    creds = authenticate_gdrive()

    # Initialize GoogleDriveLoader
    loader = GoogleDriveLoader(
        credentials=creds,
        folder_id='your-google-drive-folder-id'  # Replace with your folder ID
    )

    # Load documents from Google Drive
    documents = loader.load()

    for doc in documents:
        print(doc)

if __name__ == "__main__":
    main()

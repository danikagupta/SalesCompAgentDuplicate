# Flow of Execution

1) User Input and Initial Classifier:
The user input is passed to the graph.stream method.
The initial_classifier method is invoked since the entry point is set to "classifier".

2) Returning from Initial Classifier:
The initial_classifier method classifies the input and returns a state with the category.
The returned state includes "category": category, where category could be "policy", "commission", etc.

3) Routing in Main Router:
The main_router method receives the state and returns the category, which corresponds to the next node in the state graph.

4) StateGraph Handles Transitions:
The StateGraph framework automatically transitions to the node corresponding to the returned category.
It then calls the method associated with that node.

# Why do we need both initial_classifier and main_router?

initial_classifier: Focuses solely on classifying the user input. It doesn't decide what to do next.

main_router: Takes the classification result and determines the next state in the graph. 

This decouples the classification logic from the routing logic.

# RAG script

1) RAG script is in the file called _rag.py

2) When you run the script, it will authenticate with Google Drive using the client_secrets.json file, download the specified files from Google Drive, convert them to text, generate embeddings, and upload those embeddings to Pinecone.

3) The script will output a message in the Streamlit app indicating that the documents have been processed and embeddings uploaded to Pinecone. You can monitor the progress in your terminal or the Streamlit web interface.

4) You should see something like this in your Streamlit app: Documents processed and embeddings uploaded to Pinecone.

# How to obtain 'client_secrets.json'

1) Create a Project in Google Cloud Console:

Go to the Google Cloud Console.
Create a new project or select an existing project.

2) Enable Google Drive API:

In the Cloud Console, navigate to APIs & Services > Library.
Search for "Google Drive API" and click Enable.

3) Create OAuth 2.0 Credentials:

Go to APIs & Services > Credentials.
Click on Create Credentials > OAuth 2.0 Client IDs.
Set the Application type to "Desktop app" or "Web application" depending on your use case.
Fill in the required fields and click Create.

4) Download client_secrets.json:

After creating the credentials, download the client_secrets.json file.
Save this file in the root directory of your project.
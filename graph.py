from langchain_openai import ChatOpenAI


class salesCompAgent():
    def __init__(self, api_key):
        self.model = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0, api_key=api_key)
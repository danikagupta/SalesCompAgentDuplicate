import streamlit as state
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.create_llm_message import create_llm_message

def __init__(self, model):
    self.model = model

def clarify_and_classify(self, user_query: str) -> str:
    clarify_prompt = f"""
    You are a sales comp agent and you do xyz.
    """
    
    llm_messages = create_llm_message(clarify_prompt)
    llm_response = self.model.invoke(llm_messages)
    full_response = llm_response.content
    return full_response

def clarify_agent(self, state: dict) -> dict:
    full_response = self.clarify_and_classify(state['initialMessage'])
    
    return {
        "lnode": "clarify_agent",
        "responseToUser": full_response,
        "category": "clarify"
    }
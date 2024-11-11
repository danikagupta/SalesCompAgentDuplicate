import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def create_llm_message(system_prompt, sessionHistory):
    #print(f"CREATELLM: sessionHistory is {sessionHistory}")
    #st.write(f"CREATELLM: sessionHistory is {sessionHistory}")
    #msgs=st.session_state.messages
    #print(f"CREATELLM  msgs is {msgs}")
    resp = []
    resp.append(SystemMessage(content=system_prompt))
    resp.extend(sessionHistory)
    #print(f"CREATELLM: resp is {resp}")
    return resp

def create_llm_message(system_prompt):
    # Initialize empty list to store messages
    resp = []
    
    # Add system prompt as the first message. This will provide the overall instructions to LLM.
    resp.append(SystemMessage(content=system_prompt))
    
    # Get chat history from Streamlit's session state
    msgs = st.session_state.messages
    
    # Iterate through chat history, and based on the role (user or assistant) tag it as HumanMessage or AIMessage
    for m in msgs:
        if m["role"] == "user":
            # Add user messages as HumanMessage
            resp.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            # Add assistant messages as AIMessage
            resp.append(AIMessage(content=m["content"]))
    
    # Return the formatted message list
    return resp

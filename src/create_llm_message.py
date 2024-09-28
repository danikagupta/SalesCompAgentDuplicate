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


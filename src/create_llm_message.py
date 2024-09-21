import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def create_llm_message(system_prompt):
    msgs=st.session_state.messages
    print(f"CREATELLM  msgs is {msgs}")
    resp = []
    #print(f"session state is {st.session_state}")
    resp.append(SystemMessage(content=system_prompt))
    for m in msgs:
        if m['role'] == 'user':
            resp.append(HumanMessage(content=m['content']))
        elif m['role'] == 'assistant':
            resp.append(AIMessage(content=m['content']))
    abcd=""" 
    if 'messages' in sessionState:
        for m in sessionState.messages:
            if m['role'] == 'user':
                resp.append(HumanMessage(content=m['content']))
            elif m['role'] == 'assistant':
                resp.append(AIMessage(content=m['content']))
    """
    #resp.append(HumanMessage(content=human_prompt))
    print(f"resp is {resp}")
    return resp


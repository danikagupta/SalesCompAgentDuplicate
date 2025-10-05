import streamlit as st
from langsmith import Client
from langchain_core.callbacks import BaseCallbackHandler

class LangsmithRunRecorder(BaseCallbackHandler):
    def __init__(self):
        self.root_run_id = None

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id, **kwargs):
        if parent_run_id is None and self.root_run_id is None:
            self.root_run_id = str(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id, **kwargs):
        if parent_run_id is None and self.root_run_id is None:
            self.root_run_id = str(run_id)



def record_langsmith_feedback(feedback_id: int, feedback_value: str) -> None:
    run_id = st.session_state.get("last_langsmith_run_id")
    if not run_id:
        print("No LangSmith run id available for feedback; skipping logging.")
        return

    try:
        client = Client()
    except Exception as exc:
        print(f"Unable to initialize LangSmith client: {exc}")
        return

    
    source_info = {"source": "streamlit_feedback"}
    thread_id = st.session_state.get("thread_id")
    if thread_id:
        source_info["thread_id"] = thread_id

    try:
        client.create_feedback(
            run_id=run_id,
            key="user_feedback",
            score=feedback_id,
            value=feedback_value,
            source_info=source_info,
        )
        print(f"XXX: created feedback: {run_id=}, {feedback_id=}, {feedback_value=}")
    except Exception as exc:
        print(f"Failed to log LangSmith feedback: {exc}")

def record_feedback():

    print(f"In record_feedback")
    #print(f"{st.session_state}")
    print(f"ABCD\n\n{st.session_state.get('feedback_id')}")
    feedback_id = st.session_state.get('feedback_id')
    feedback_value = "positive" if feedback_id > 0 else "negative"
    record_langsmith_feedback(feedback_id, feedback_value)

def accept_feedback():
    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs", on_change=record_feedback, key="feedback_id")
    if selected is not None:
        st.markdown(f"You selected: {sentiment_mapping[selected]}")
        record_langsmith_feedback(sentiment_mapping[selected])


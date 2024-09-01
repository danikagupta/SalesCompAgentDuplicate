import streamlit as st

def show_navigation():
    with st.container(border=True):
        st.page_link("pages/upload_pdf.py", label="Upload PDF", icon="1️⃣")
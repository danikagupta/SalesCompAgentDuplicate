import streamlit as st

def show_navigation():
    with st.container(border=True):
        col1,col2,col3=st.columns(3)
        col1.page_link("streamlit_app.py", label="Home", icon="🏠")
        col2.page_link("pages/0_upload_pdf.py", label="Upload PDF", icon="1️⃣")
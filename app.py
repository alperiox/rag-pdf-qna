import os

import streamlit as st
from dotenv import load_dotenv

from chatbot import get_conversational_rag_chain, set_up_vectordb
from process_data import process_pdf

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


with st.sidebar:
    api_key = st.text_input("GCP API Key", key="chatbot_api_key", type="password", value=api_key)
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

st.title("💬 Talk with PDFs!")

uploaded_file = st.file_uploader("Upload a PDF", type=("pdf"))
# set up the vector database 
if uploaded_file:
    if not api_key:
        st.info("Please add your GCP API key to continue.")
        st.stop()

    bar = st.progress(0, text="Processing PDF...")
    docs = process_pdf(uploaded_file)
    bar.progress(33.0, text="Setting up Chroma Vector DB...")
    retriever = set_up_vectordb(docs)
    bar.progress(66.0, text="Setting up the conversational RAG chain...")
    # set up the conversational RAG chain
    rag_chain = get_conversational_rag_chain(retriever)
    bar.progress(100.0, text="Done!")


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you?"}]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="Can you give me a short summary?", disabled = not uploaded_file):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # get the response from the chain 
    response = rag_chain.invoke(
        {
            "input": prompt
        },
        config={"configurable": {"session_id": "0"}},
    )['answer']

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
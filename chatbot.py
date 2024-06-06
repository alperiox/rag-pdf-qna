import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prompts import contextualize_q_prompt, qa_prompts

load_dotenv()
llm = ChatVertexAI(model="gemini-pro")
history = ChatMessageHistory()
embds = VertexAIEmbeddings(model_name="textembedding-gecko")


def set_up_vectordb(dbname: str, docs: list[Document] = None):
    # check if the database exists
    if os.path.exists(os.path.join("vectordbs", dbname)):
        vectorstore = Chroma(
            persist_directory=os.path.join("vectordbs", dbname),
            embedding_function=embds,
        )
        retriever = vectorstore.as_retriever()
        return retriever
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embds,
            persist_directory=os.path.join("vectordbs", dbname),
        )
        retriever = vectorstore.as_retriever()
        return retriever


def get_conversational_rag_chain(vdb_retriever):
    history_aware_retriever = create_history_aware_retriever(
        llm, vdb_retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompts)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda sess_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

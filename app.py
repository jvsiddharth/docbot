import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    load_dotenv()
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorestore):
    load_dotenv()
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_hist', return_messasges=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="DOCBOT : Train GPT on your PDF", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("DOCBOT : Train GPT on your PDF :books:")
    st.text_input("Ask him anythin about that PDF:")

    with st.sidebar:
        st.subheader("Your PDF")
        pdf_docs = st.file_uploader(
            "Upload", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
            #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)

            #get text chunk
            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)

            #create vector store
            vectorstore = get_vectorstore(text_chunks)
            st.write(vectorstore)

            #conversation chain
            conversation = get_conversation_chain(vectorstore)
            st.write(conversation)

    st.session_state.conversation


if __name__ == '__main__':
    main()
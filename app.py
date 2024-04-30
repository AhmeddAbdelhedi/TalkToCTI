import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
# Load your OpenAI API key 
os.environ["OPENAI_API_KEY"] = "YOURKEYHERE"

# Pre-load the PDF document 
loader = PyPDFLoader("./ref.pdf")
document = loader.load()

# Pre-process the document (split into chunks)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=75)
texts = text_splitter.split_documents(document)

# Create the LLM, embedding, and vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# Create the retriever for document search
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the QA chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)


# Streamlit App

st.title("Docker Cryptomining Guru")
st.markdown("In the day we shine, in the night we mine. How can I enlighten your cryptomining journey?")

# Display a summary of the report (optional)
with st.expander("Report Summary"):
    st.write("This report provides information on Docker-based cryptomining. It covers aspects like...")
 

# Chat interface
user_query = st.chat_input(placeholder="Ask a question about the report")

if user_query is not None:
    with st.spinner("Answering your question..."):
        result = qa({"query": user_query})

    st.write(f"**Your Question:** {user_query}")
    st.write(f"**Answer:** {result['result']}")


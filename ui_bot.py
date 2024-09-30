import os
import openai
import streamlit as st
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
import io
from langchain.vectorstores.cassandra import Cassandra
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio

# Load environment variables for secrets
from dotenv import load_dotenv
load_dotenv()

# Set up API keys and DB credentials
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')

openai.api_key = OPENAI_API_KEY

# Initialize Azure Blob Client


def load_pdf_from_azure():
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(
        container='ragpdf-container', blob='what_is_science.pdf')
    download_stream = blob_client.download_blob()
    return download_stream.readall()

# Extract text from the PDF


def extract_text_from_pdf(pdf_content):
    pdf_stream = io.BytesIO(pdf_content)
    pdf_reader = PdfReader(pdf_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Initialize Astra DB connection


def init_astra_db():
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create a vector store in Astra DB


def create_vectorstore(embedding):
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="qa_pdf_demo",
        session=None,
        keyspace=None,
    )
    return astra_vector_store

# Main App


def main():
    st.title("PDF QA with Azure and Astra DB")

    # Initialize session state variables
    if 'pdf_loaded' not in st.session_state:
        st.session_state.pdf_loaded = False
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    if 'astra_vector_store' not in st.session_state:
        st.session_state.astra_vector_store = None
    if 'astra_vector_index' not in st.session_state:
        st.session_state.astra_vector_index = None

    # Step 1: Load PDF UI (only show this initially)
    if not st.session_state.pdf_loaded:
        if st.button("Load PDF from Azure"):
            st.session_state.pdf_content = load_pdf_from_azure()
            st.session_state.extracted_text = extract_text_from_pdf(
                st.session_state.pdf_content)
            st.success("PDF Loaded and Text Extracted!")

            # Initialize Astra DB connection and process text
            init_astra_db()

            # Set up the OpenAI embedding model
            embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

            # Split text for chunking
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=800, chunk_overlap=200, length_function=len
            )
            texts = text_splitter.split_text(st.session_state.extracted_text)

            # Store text chunks into Astra DB vector store
            st.session_state.astra_vector_store = create_vectorstore(embedding)
            st.session_state.astra_vector_store.add_texts(texts)
            st.success(f"Inserted {len(texts)} chunks into Astra DB.")

            # Set up vector index
            st.session_state.astra_vector_index = VectorStoreIndexWrapper(
                vectorstore=st.session_state.astra_vector_store
            )

            # Mark PDF as loaded
            st.session_state.pdf_loaded = True

    # Step 2: QA interface (show only after PDF is loaded)
    if st.session_state.pdf_loaded:
        st.write("PDF Loaded! You can now ask questions.")

        # Question-answering loop
        user_question = st.text_input("Ask a question:")
        if st.button("Get Answer") and user_question:
            if st.session_state.astra_vector_index:
                # Query the vector store using OpenAI LLM
                llm = OpenAI(openai_api_key=OPENAI_API_KEY)
                answer = st.session_state.astra_vector_index.query(
                    user_question, llm=llm).strip()
                st.write("Answer:", answer)

                # Display top relevant documents
                st.write("Top relevant documents:")
                for doc, score in st.session_state.astra_vector_store.similarity_search_with_score(user_question, k=4):
                    st.write(f"[{score:.4f}] {doc.page_content[:100]}...")
            else:
                st.warning("Please load the PDF and process it first.")


if __name__ == "__main__":
    main()

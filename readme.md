# PDF Question-Answering System with Azure and Astra DB

This project is a demo application that enables question-answering from PDF documents stored in Azure Blob Storage, powered by OpenAI embeddings and vector search in Astra DB. The app allows users to upload a PDF, store its contents in Astra DB as vectorized chunks, and ask questions to retrieve relevant answers from the document.

## Features

- **Azure Blob Storage**: Load PDFs stored in an Azure Blob Storage account.
- **Text Extraction**: Extract text from PDF documents using `PyPDF2`.
- **Vector Store**: Store the extracted text chunks in Astra DB, powered by the `Cassandra` vector store from LangChain.
- **Question-Answering**: Use OpenAI's GPT to answer questions based on the vectorized document chunks.
- **Vector Search**: Retrieve the most relevant text chunks from Astra DB for answering the user's questions.

## Prerequisites

Before running the application, ensure you have the following:

1. **Azure Blob Storage**: A PDF file uploaded to an Azure Blob Storage container.
2. **Astra DB Account**: A Serverless Cassandra with Vector Search database on [Astra DB](https://astra.datastax.com).
3. **OpenAI API Key**: Access to OpenAI's API for embeddings and language model responses.

### Required API Keys and Connection Details:

- **Azure Connection String**: Your Azure Storage account connection string.
- **Astra DB Token**: A token with the `Database Administrator` role from Astra DB.
- **OpenAI API Key**: Your OpenAI API key for embeddings and LLM capabilities.

## Setup

### Step 1: Clone the Repository

Clone the repository to your local machine.

### Step 2: Install Dependencies

Ensure you have the necessary Python packages installed. Use a package manager like `pip` to install the required libraries specified in `requirements.txt`.

### Step 3: Configure Environment Variables

Create a `.env` file in the root of the project directory and add your configuration details:

OPENAI_API_KEY=your_openai_api_key
AZURE_CONNECTION_STRING=your_azure_connection_string ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
ASTRA_DB_ID=your_astra_db_id


## Usage

1. **Run the Application**: Start the Streamlit application by executing the main script. This typically involves running a command in your terminal (e.g., `streamlit run app.py`, where `app.py` is the name of your main script).

2. **Load PDF**: Click the "Load PDF from Azure" button to retrieve the PDF file from Azure Blob Storage.

3. **Ask Questions**: Once the PDF is loaded, you can enter your questions in the text input field to get answers based on the content of the document.

4. **Review Relevant Documents**: The application will display the top relevant documents from Astra DB that were used to formulate the answer.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Azure Blob Storage](https://azure.microsoft.com/en-us/services/storage/blobs/)
- [Astra DB](https://astra.datastax.com)
- [OpenAI](https://openai.com)
- [LangChain](https://langchain.readthedocs.io/en/latest/)

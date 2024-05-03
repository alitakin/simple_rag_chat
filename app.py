import os
import boto3
import streamlit as st
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import faiss
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS and FAISS Configuration
s3_client = boto3.client('s3')
bucket_name = "mahdiyehfaisisbucket"
folder_name = "faiss_index"  # S3 folder name

# Local directory for FAISS files
local_directory = "faiss_index"
os.makedirs(local_directory, exist_ok=True)  # Ensure local directory exists

# File names in the S3 bucket
index_files = ["index.faiss", "index.pkl"]

# Initialize Bedrock client and embeddings
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def download_file_from_s3(key):
    """Download a specific file from S3 to the local directory if it does not already exist."""
    local_file_path = os.path.join(local_directory, key)
    if not os.path.exists(local_file_path):
        try:
            s3_client.download_file(bucket_name, f"{folder_name}/{key}", local_file_path)
            st.write(f"Downloaded {key} to {local_file_path}")
        except ClientError as e:
            st.error(f"Failed to download {key}: {e}")
            raise

def get_vector_store():
    """Ensure all required FAISS index files are present locally, downloading from S3 if necessary."""
    for file_name in index_files:
        download_file_from_s3(file_name)
    
    # Load the FAISS index file after ensuring it is present
    index_file_path = os.path.join(local_directory, "index.faiss")
    return faiss.read_index(index_file_path)  # Load FAISS index


def get_llama2_llm():
    return Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})



prompt_template = """
Human: Use the following context to provide a concise answer but summarize with 250 words with detailed explanations. If you don't know the answer, just say that you don't know.
<context>{context}</context>
Question: {question}
Assistant:"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']
import streamlit as st
import streamlit as st

def main():
    st.set_page_config(page_title="Style Advisor: Interactive Fashion Chat")
    st.header("Fashion Assistant💁: Powered by AWS Bedrock")

    # Consolidate custom CSS
    custom_css = """
    <style>
    #GithubIcon {
        visibility: hidden;
    }
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    .footer {
        font-size: 16px;
        font-weight: bold;
        color: #2E2E2E;
        text-align: center;
        padding: 10px;
        position: fixed;
        bottom: 0;
        left: 0;    /* Ensures the footer starts from the far left */
        width: 100%;    /* Ensures the footer extends full width */
        background-color: transparent;
        z-index: 100;  /* Ensures the footer stays on top of other elements */
    }
    /* Additional style to ensure Streamlit's own footer is not interfering */
    .css-1wrcr25 {
        display: none; /* This class name might change with Streamlit updates */
    }
    </style>
    <div class="footer">
        Developed by Mahdiyeh Sadat Mohajeri with ❤️ | Centria University of Applied Sciences
    </div>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Download and Load Vector Store"):
            with st.spinner("Downloading and Loading Vector Store..."):
                faiss_index = get_vector_store()  # Assume this is correctly defined
                st.success("FAISS index loaded successfully!")

    if st.button("Process Query"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()  # Assume this is correctly defined
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)
            st.success("Query processed successfully!")

if __name__ == "__main__":
    main()

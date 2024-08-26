
simple_rag_chat [WIP]
===============

A simple RAG (Retrieval-Augmented Generation) chatbot utilizing Amazon Bedrock for backend processing and FAISS for efficient retrieval from a local dataset.

Key Features
------------

- **Amazon Bedrock**: Leverage Amazon's infrastructure for scalable, secure model deployment.
- **FAISS Integration**: Enables quick and effective similarity search.
- **Streamlit UI**: Provides a simple web interface for interacting with the chatbot.

Installation
------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/alitakin/simple_rag_chat.git

2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

Running the Application
-----------------------

Start the Streamlit app:

.. code-block:: bash

   streamlit run app.py

Directory Structure
-------------------

- ``app.py``: Main application code.
- ``data/``: Contains the datasets used by the chatbot.
- ``faiss_index/``: Directory for FAISS index files.



License
-------

This project is licensed under the MIT License.

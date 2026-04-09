import os
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.combine_docs import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load documents from the specified directory
def load_documents(directory):
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PDFPlumberLoader)
    return loader.load()

# Split documents into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# create embeddings
def create_and_store_embeddings(chunks, vectorstore_path="vectorstore"):
    """Initializes the HuggingFaceEmbeddings, creates embeddings for the chunks, builds a FAISS vector store, and saves it to the vector_store/ directory."""

    embeddings = HuggingFaceEmbeddings()
    vectore_store = FAISS.from_documents(chunks, embeddings)
    vectore_store.save_local(vectorstore_path)
    
def load_retriever(vectorstore_path="vectorstore"):
    """Loads the saved FAISS index and returns it as a LangChain retriever."""
    embeddings = HuggingFaceEmbeddings()
    vectore_store = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_code_deserialization=True)
    return vectore_store.as_retriever()


def create_rag_chain(retriever, local_llm_url, api_key):
    template = """"
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {input}
    Context: {context}
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["input", "context"])
    llm = OpenAI(base_url=local_llm_url, api_key=api_key)
                 
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    return create_retrieval_chain(retriever, document_chain)

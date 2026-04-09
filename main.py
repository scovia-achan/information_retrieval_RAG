import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from core_logic import (
    load_documents, 
    split_documents, 
    create_and_store_embeddings, 
    load_retriever, 
    create_rag_chain,
) 

load_dotenv()

if __name__ == "__main__":
    vector_db_path = os.getenv("VECTOR_DB_DIR")
    data_dir = os.getenv("DATA_DIR")

    if not os.path.exists(vector_db_path) or not os.listdir(vector_db_path):
        print("Vector database not found. Creating new vector database...")
        documents = load_documents(data_dir)
        chunks = split_documents(documents)
        create_and_store_embeddings(chunks=chunks, vectorstore_path=vector_db_path)
    else:
        print("Loading existing vector database...")
        
    retriever = load_retriever(vectorstore_path=vector_db_path)
    # Use online model like Gemini instead of local LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY"))

    if not local_llm_url:
        raise ValueError("LOCAL_LLM_URL environment variable is not set.")
    rag_chain = create_rag_chain(retriever=retriever, local_llm_url=local_llm_url)
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = rag_chain.invoke({"input":query, "context":""})
        print(f"Answer: {response['answer']}\n")
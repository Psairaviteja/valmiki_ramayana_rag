from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load the embedding model and the saved Chroma database

def retreiver(user_query):
    persist_directory = "./chroma_lc_db"
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    # 3. Perform similarity search to retrieve relevant Ramayana chunks
    retrieved_docs = vector_db.similarity_search(user_query, k=10)

    return retrieved_docs

if __name__ == "__main__":
    user_query = "Who is Ravana"
    retrieved_docs = retreiver(user_query)
    print(f"User Query: '{user_query}'\n")
    print(f"Retrieved {len(retrieved_docs)} chunks")
    print(retrieved_docs)
    print("--- Retrieved Ramayana Context ---")
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print("Context:" ,context )
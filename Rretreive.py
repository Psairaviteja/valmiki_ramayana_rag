from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "./chroma_lc_db"
COLLECTION = "valmiki_ramayana"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def retreiver(user_query):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_db = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_PATH,
        embedding_function=embedding_model,
    )

    # Retrieve the top 10 most similar Ramayana chunks.
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
    print("Context:", context)

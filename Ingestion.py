"""
LangChain version: ingest bala_kanda_english.json into Chroma with metadata.

    pip install langchain-chroma langchain-huggingface langchain-text-splitters sentence-transformers
    python ingest_langchain.py
"""
import json
from itertools import islice
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

JSON_FILE = "Extracted_json/bala_kanda_english.json"
DB_PATH = "./chroma_lc_db"
COLLECTION = "valmiki_ramayana"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest():
    with open(JSON_FILE, encoding="utf-8") as f:
        chapters = {int(k): v for k, v in json.load(f).items()}

    
    # chunk_size is in characters (~1000 chars is roughly 170 words)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    # each chunk automatically inherits its chapter's metadata
    docs = splitter.create_documents(
        texts=list(chapters.values()),
        metadatas=[{"kanda": "bala", "chapter": n} for n in chapters],
    )
    print("Docs",docs[:3])
    # stable IDs so re-running overwrites instead of duplicating
    counts, ids = {}, []
    for d in docs:
        ch = d.metadata["chapter"]
        counts[ch] = counts.get(ch, 0) + 1
        ids.append(f"bala-{ch:03d}-{counts[ch]:03d}")

    print(dict(islice(counts.items(), 3)))
    # print("Counts",counts[:3])
    print("ids",ids[:3])

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Ingested {len(docs)} chunks from {len(chapters)} chapters")

# data = vector_store.get(include=["documents", "metadatas"])

# print("Total chunks:", len(data["ids"]))


def get_store():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
        # collection uses cosine distance, so relevance = 1 - distance (higher = better).
        # Set explicitly because reopening a store doesn't re-read the distance setting.
        relevance_score_fn=lambda distance: 1 - distance,
    )


def inspect_db():
    vs = get_store()

    # Get records from the underlying Chroma collection
    data = vs.get(
        include=["documents", "metadatas", "embeddings"]
    )

    print("Total chunks:", len(data["ids"]))

    for i in range(min(3, len(data["ids"]))):
        print("\nID:", data["ids"][i])
        print("Text:", data["documents"][i][:200])
        print("Metadata:", data["metadatas"][i])

        embedding = data["embeddings"][i]
        print("Embedding dimensions:", len(embedding))
        print("First 5 values:", embedding[:5])


def demo():
    vs = get_store()

    def show(results):
        """results is a list of (Document, relevance_score) tuples."""
        for d, score in results:
            print(f"[score {score:.3f} | ch {d.metadata['chapter']}] "
                  f"{d.page_content[:120]}...")
        print("==============")

    # plain semantic search with relevance scores (0-1, higher = more relevant)
    # show(vs.similarity_search_with_relevance_scores(
    #     "Rama breaks the bow of Shiva", k=3))

    # one chapter
    show(vs.similarity_search_with_relevance_scores(
        "the demoness", k=3, filter={"chapter": 24}))

    # # chapter range (one operator per condition, join with $and)
    # show(vs.similarity_search_with_relevance_scores(
    #     "sage Vishvamitra", k=3,
    #     filter={"$and": [{"chapter": {"$gte": 51}}, {"chapter": {"$lte": 65}}]},
    # ))

    # # drop weak matches: only keep results scoring at least 0.4
    # show(vs.similarity_search_with_relevance_scores(
    #     "marriage", k=5, score_threshold=0.4,
    #     filter={"chapter": {"$in": [72, 73, 77]}},
    # ))

    # # keyword filter on the text itself
    # show(vs.similarity_search_with_relevance_scores(
    #     "ritual", k=3, where_document={"$contains": "Rishyasringa"}))


if __name__ == "__main__":
    ingest()
    inspect_db()
    # demo()

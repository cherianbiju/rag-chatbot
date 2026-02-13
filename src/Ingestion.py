import os
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec

load_dotenv()



# CONFIG


class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cadtesting")
    EMBEDDING_MODEL = "microsoft/codebert-base"
    EMBEDDING_DIMENSION = 768
    DOCUMENTS_DIR = Path("../documents")



# METADATA EXTRACTION


def extract_metadata(code: str, filename: str) -> Dict:
    all_ops = []

    for op in [
        "draw", "drawCircle", "extrude", "revolve",
        "fillet", "shell", "cut", "fuse",
        "loft", "sweep", "translate", "rotate"
    ]:
        if op in code:
            all_ops.append(op)

    patterns = []
    lower_code = code.lower()

    if "shell" in lower_code:
        patterns.append("hollow")
    if "revolve" in lower_code:
        patterns.append("revolution")
    if "fillet" in lower_code or "chamfer" in lower_code:
        patterns.append("rounded")
    if "cut" in lower_code or "fuse" in lower_code:
        patterns.append("boolean")

    complexity = len(all_ops) * 10
    if "loft" in code or "sweep" in code:
        complexity += 20

    complexity = min(complexity, 100)

    return {
        "filename": filename,
        "operations": ",".join(all_ops[:10]),
        "patterns": ",".join(patterns),
        "complexity": complexity
    }


# LOAD DOCUMENTS 


def load_documents(docs_dir: Path) -> List[Document]:
    print("\nLoading CAD model files only...\n")

    documents = []

    for file_path in sorted(docs_dir.iterdir()):

        # Only allow JavaScript CAD files
        if file_path.suffix == ".js":

            # Extra safety: skip any system-like files
            if "system" in file_path.name.lower():
                print(f"Skipped (system file): {file_path.name}")
                continue

            code = file_path.read_text(encoding="utf-8")
            metadata = extract_metadata(code, file_path.name)

            documents.append(
                Document(
                    text=code,
                    metadata=metadata
                )
            )

            print(f"Stored: {file_path.name}")

        else:
            print(f"Skipped (not CAD model): {file_path.name}")

    print(f"\nTotal CAD documents stored: {len(documents)}\n")
    return documents



# CHUNK DOCUMENTS 


def chunk_documents(documents: List[Document]) -> List[TextNode]:
    print("Preparing vectors...\n")

    chunks = []

    for doc in documents:
        metadata = doc.metadata.copy()
        metadata["is_complete_model"] = True

        node = TextNode(
            text=doc.text,
            metadata=metadata
        )

        chunks.append(node)
        print(f"{metadata.get('filename')} → 1 vector")

    print(f"\nTotal vectors created: {len(chunks)}\n")
    return chunks



# SETUP PINECONE


def setup_pinecone(force_recreate=False):
    print("Setting up Pinecone...\n")

    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    if force_recreate and Config.PINECONE_INDEX_NAME in pc.list_indexes().names():
        print("Deleting existing index...")
        pc.delete_index(Config.PINECONE_INDEX_NAME)
        time.sleep(3)

    if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index...")
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=Config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

        while not pc.describe_index(Config.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    print("Pinecone index ready.\n")
    return pc.Index(Config.PINECONE_INDEX_NAME)



# CREATE EMBEDDINGS & STORE


def store_embeddings(chunks: List[TextNode], pinecone_index):
    print("Creating CodeBERT embeddings...\n")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBEDDING_MODEL
    )

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Processing {len(chunks)} vectors...\n")

    index = VectorStoreIndex(
        nodes=chunks,
        storage_context=storage_context,
        show_progress=True
    )

    print(f"\nSuccessfully stored {len(chunks)} vectors.\n")
    return index



# MAIN BUILD FUNCTION


def build_rag(force_recreate=False):

    documents = load_documents(Config.DOCUMENTS_DIR)
    chunks = chunk_documents(documents)
    pinecone_index = setup_pinecone(force_recreate)
    store_embeddings(chunks, pinecone_index)

    print("✅ RAG BUILD COMPLETE\n")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    build_rag(force_recreate=force)

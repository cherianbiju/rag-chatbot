import os
import re
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cadtesting")
    EMBEDDING_MODEL = "microsoft/codebert-base"
    EMBEDDING_DIMENSION = 768
    DOCUMENTS_DIR = Path("../documents")



# METADATA EXTRACTION

def extract_metadata(code: str, filename: str) -> Dict:
    all_ops = []
    for op in ["draw", "drawCircle", "extrude", "revolve", "fillet", "shell", 
               "cut", "fuse", "loft", "sweep", "translate", "rotate"]:
        if op in code:
            all_ops.append(op)
    
    patterns = []
    if "shell" in code.lower():
        patterns.append("hollow")
    if "revolve" in code.lower():
        patterns.append("revolution")
    if "fillet" in code.lower() or "chamfer" in code.lower():
        patterns.append("rounded")
    if "cut" in code.lower() or "fuse" in code.lower():
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
    print("\n Loading documents...")
    
    documents = []
    
    for file_path in sorted(docs_dir.iterdir()):
        if file_path.suffix == ".js":
            code = file_path.read_text(encoding="utf-8")
            metadata = extract_metadata(code, file_path.name)
            documents.append(Document(text=code, metadata=metadata))
            print(f"{file_path.name} (ops: {len(metadata['operations'].split(','))})")
        
        elif file_path.suffix in [".txt", ".md"]:
            text = file_path.read_text(encoding="utf-8")
            metadata = {"filename": file_path.name, "file_type": "text"}
            documents.append(Document(text=text, metadata=metadata))
            print(f"{file_path.name}")
    
    print(f"\n Total: {len(documents)} documents\n")
    return documents


# CHUNK DOCUMENTS

def chunk_documents(documents: List[Document]) -> List[TextNode]:
    print("Preparing documents ...")

    chunks = []

    for doc in documents:
        metadata = doc.metadata.copy()
        metadata["is_complete_model"] = True

        node = TextNode(
            text=doc.text,
            metadata=metadata
        )

        chunks.append(node)
        print(f"{metadata.get('filename', 'unknown')} → 1 vector")

    print(f"\n Total: {len(chunks)} vectors\n")
    return chunks



# SETUP PINECONE

def setup_pinecone(force_recreate=False):
    print("Setting up Pinecone...")
    
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # Delete 
    if force_recreate and Config.PINECONE_INDEX_NAME in pc.list_indexes().names():
        print(f"Deleting old index...")
        pc.delete_index(Config.PINECONE_INDEX_NAME)
        time.sleep(3)
    
    if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index...")
        pc.create_index(
            name=Config.PINECONE_INDEX_NAME,
            dimension=Config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(Config.PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
    
    index = pc.Index(Config.PINECONE_INDEX_NAME)
    print(f"Index ready: {Config.PINECONE_INDEX_NAME}\n")
    return index


# CREATE EMBEDDINGS & STORE

def store_embeddings(chunks: List[TextNode], pinecone_index):
    print("Creating CodeBERT embeddings...")
    
    # Setup CodeBERT
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBEDDING_MODEL
    )
    
    # Create vector store and index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"Processing {len(chunks)} chunks...")
    
    index = VectorStoreIndex(
        nodes=chunks,
        storage_context=storage_context,
        show_progress=True
    )
    
    print(f"\n Stored {len(chunks)} vectors\n")
    return index


# MAIN

def build_rag(force_recreate=False):
    
    # Load documents
    documents = load_documents(Config.DOCUMENTS_DIR)
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Setup Pinecone
    pinecone_index = setup_pinecone(force_recreate)
    
    # Store embeddings
    store_embeddings(chunks, pinecone_index)
    
    print("  ✅ BUILD COMPLETE")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    build_rag(force_recreate=force)
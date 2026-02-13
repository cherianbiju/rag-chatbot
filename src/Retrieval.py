import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone
import time
from typing import List

load_dotenv()


# CONFIGURATION


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cadtesting")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "models/gemini-3-flash-preview"
EMBEDDING_MODEL = "microsoft/codebert-base"

SYSTEM_PROMPT_PATH = "../documents/replicad_system_prompt.txt"


# CUSTOM RETRIEVER


class HybridFilenameRetriever(VectorIndexRetriever):
    
    def __init__(self, index, similarity_top_k=20, filename_boost=1.5, **kwargs):
        super().__init__(index=index, similarity_top_k=similarity_top_k, **kwargs)
        self.filename_boost = filename_boost
        self.final_top_k = 1
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        
        query_lower = query_bundle.query_str.lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]
        
        print(f"\nQuery words: {query_words}")
        
        nodes = super()._retrieve(query_bundle)
        print(f"Initial retrieval: {len(nodes)} nodes")
        
        boosted_nodes = []
        
        for node in nodes:
            filename = node.metadata.get("filename", "").lower()
            matches = sum(1 for word in query_words if word in filename)
            
            if matches > 0:
                boost = self.filename_boost * matches
                node.score += boost
                print(f"   🎯 BOOSTED '{filename}': +{boost:.2f} → score: {node.score:.3f}")
            
            boosted_nodes.append(node)
        
        boosted_nodes.sort(key=lambda x: x.score, reverse=True)
        
        print(f"\nFinal ranking (top {self.final_top_k}):")
        for i, node in enumerate(boosted_nodes[:self.final_top_k], 1):
            filename = node.metadata.get("filename", "Unknown")
            print(f"   {i}. {filename} (score: {node.score:.3f})")
        
        return boosted_nodes[:self.final_top_k]



# SETUP


def setup():
    
    print("Setting up RAG system...")
    
    # LOAD SYSTEM PROMPT
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    # INJECT SYSTEM PROMPT INTO GEMINI
    Settings.llm = GoogleGenAI(
        model=LLM_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.1,
        request_timeout=300,
        max_retries=3,
        system_prompt=system_prompt   
    )
    
    print("Loading CodeBERT...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    retriever = HybridFilenameRetriever(
        index=index,
        similarity_top_k=20,
        filename_boost=1.5
    )
    
    query_engine = RetrieverQueryEngine(retriever=retriever)
    
    print("✅ RAG ready\n")
    return query_engine


try:
    query_engine = setup()
except Exception as e:
    print(f"❌ Setup failed: {e}")
    query_engine = None



# QUERY


def query_with_retry(question: str, max_retries=2):
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry {attempt+1}...")
                time.sleep(5)
            return query_engine.query(question)
        except Exception as e:
            if "504" in str(e) and attempt < max_retries - 1:
                continue
            raise


def query(question: str):
    
    if not query_engine:
        print("System not initialized")
        return
    
    print("\n" + "="*80)
    print(f"QUERY: {question}")
    print("="*80)
    
    try:
        start = time.time()
        response = query_with_retry(question)
        elapsed = time.time() - start
        
        print(f"\nCompleted in {elapsed:.1f}s\n")
        print("="*80)
        print("🤖 GENERATED CODE")
        print("="*80)
        print(response.response)
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("\n" + "="*80)
            print("SOURCES USED")
            print("="*80)
            for i, node in enumerate(response.source_nodes, 1):
                print(f"\n{i}. {node.metadata.get('filename', 'Unknown')} (score: {node.score:.3f})")
        
        return response
    
    except Exception as e:
        print(f"\nError !!! : {e}")
        return None


# MAIN


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query(" ".join(sys.argv[1:]))
    else:
        print("\nType 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("Question: ").strip()
                if not user_input or user_input.lower() in ['exit', 'quit']:
                    print("\nEnding the querying!")
                    break
                query(user_input)
            except KeyboardInterrupt:
                print("\nEnding the querying!")
                break

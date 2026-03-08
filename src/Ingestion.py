"""
ingest.py — CAD Code Ingestion Pipeline
Embeds Replicad code examples using Gemini
and stores vectors in Pinecone.
"""

import os
import sys
import time
import hashlib
import argparse
from pathlib import Path
from typing import Optional, List

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ─────────────────────────────
# CONFIG
# ─────────────────────────────

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CODE_INDEX_NAME = os.getenv("CODE_INDEX_NAME", "cad-code-examples")

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072

PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

UPSERT_BATCH_SIZE = 50
EMBED_DELAY = 0.3

DEFAULT_CODE_DIR = r"C:\Users\ASUS\Desktop\RAG_testing\code_data"

client = None


# ─────────────────────────────
# GEMINI
# ─────────────────────────────

def init_gemini():
    global client

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY missing")

    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"✅ Gemini ready ({EMBEDDING_MODEL})")

    emb = get_embedding("connection test")

    if emb is None:
        raise RuntimeError("Embedding test failed")

    print(f"✅ Embedding working ({len(emb)} dims)")


def get_embedding(text: str, retries: int = 5) -> Optional[List[float]]:

    for attempt in range(retries):

        try:

            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM
                )
            )

            return list(response.embeddings[0].values)

        except Exception as e:

            err = repr(e)

            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 20 * (attempt + 1)
                print(f"Rate limited → wait {wait}s")
                time.sleep(wait)
                continue

            if "503" in err or "UNAVAILABLE" in err:
                wait = 10 * (attempt + 1)
                print(f"Server busy → wait {wait}s")
                time.sleep(wait)
                continue

            print("Embedding error:", err)
            time.sleep(5)

    return None


# ─────────────────────────────
# PINECONE
# ─────────────────────────────

def init_pinecone():

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [i.name for i in pc.list_indexes()]

    if CODE_INDEX_NAME not in existing:

        print("Creating Pinecone index...")

        pc.create_index(
            name=CODE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

        while not pc.describe_index(CODE_INDEX_NAME).status["ready"]:
            print("Waiting for index...")
            time.sleep(3)

        print("✅ Index created")

    else:
        print("✅ Using existing index")

    return pc.Index(CODE_INDEX_NAME)


# ─────────────────────────────
# METADATA
# ─────────────────────────────

def build_metadata(filepath: Path, code: str):

    lines = code.count("\n")

    complexity = "simple"

    if lines > 150:
        complexity = "complex"
    elif lines > 60:
        complexity = "medium"

    return {
        "file": filepath.name,
        "lines": lines,
        "complexity": complexity,
        "code": code
    }


# ─────────────────────────────
# EMBEDDING TEXT
# ─────────────────────────────

def build_embedding_text(name, metadata, code):

    return f"""
Replicad CAD example: {name}

Complexity: {metadata['complexity']}
Lines: {metadata['lines']}

Code:

{code}
"""


# ─────────────────────────────
# INGEST
# ─────────────────────────────

def ingest(code_dir, dry_run=False):

    code_path = Path(code_dir)

    files = sorted(code_path.rglob("*.md"))

    if not files:
        print("No files found")
        return

    print(f"Found {len(files)} files")

    init_gemini()

    index = None
    if not dry_run:
        index = init_pinecone()

    vectors = []

    success = 0
    errors = 0

    for i, file in enumerate(files, 1):

        print(f"[{i}/{len(files)}] {file.name}")

        try:

            code = file.read_text(encoding="utf-8")

            metadata = build_metadata(file, code)

            embed_text = build_embedding_text(file.stem, metadata, code)

            embedding = get_embedding(embed_text)

            if embedding is None:
                errors += 1
                continue

            vector_id = hashlib.md5(
                str(file).encode()
            ).hexdigest()

            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })

            success += 1

            if len(vectors) >= UPSERT_BATCH_SIZE and not dry_run:

                index.upsert(vectors=vectors)

                print(f"Upserted {len(vectors)}")

                vectors = []

            time.sleep(EMBED_DELAY)

        except Exception as e:

            print("Error:", repr(e))

            errors += 1

    if vectors and not dry_run:
        index.upsert(vectors=vectors)

    print("\nFinished")
    print("Success:", success)
    print("Errors:", errors)

    if index:
        stats = index.describe_index_stats()
        print("Total vectors:", stats["total_vector_count"])


# ─────────────────────────────
# MAIN
# ─────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", default=DEFAULT_CODE_DIR)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    try:

        ingest(args.dir, args.dry_run)

    except KeyboardInterrupt:
        sys.exit(1)

    except Exception as e:

        print("Fatal:", repr(e))
        sys.exit(1)
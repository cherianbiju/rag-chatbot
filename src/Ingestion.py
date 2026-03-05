"""
ingest.py — CAD Code Ingestion Pipeline
========================================
Reads .md Replicad code example files, embeds them with
Microsoft CodeBERT, and upserts into Pinecone.
"""

import os
import sys
import time
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
CODE_INDEX_NAME   = os.getenv("CODE_INDEX_NAME", "cad-code-examples")

CODEBERT_MODEL    = "microsoft/codebert-base"
EMBEDDING_DIM     = 768   # CodeBERT CLS token output dim

PINECONE_METRIC   = "cosine"
PINECONE_CLOUD    = "aws"
PINECONE_REGION   = "us-east-1"

UPSERT_BATCH_SIZE = 50

DEFAULT_CODE_DIR  = r"C:\Users\ASUS\Desktop\RAG_testing\code_data"

tokenizer = None
model     = None
device    = None


# ─────────────────────────────────────────────
# CODEBERT SETUP
# ─────────────────────────────────────────────

def init_codebert():
    global tokenizer, model, device

    print(f"  Loading CodeBERT: {CODEBERT_MODEL} ...")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    model     = AutoModel.from_pretrained(CODEBERT_MODEL).to(device)
    model.eval()

    print(f"  ✅ CodeBERT loaded on {device}")

    # Smoke test
    print("  🔬 Testing embedding...")
    test = get_embedding("test connection")
    if test is None:
        raise RuntimeError("Embedding smoke test failed")
    print(f"  ✅ Smoke test passed — got {len(test)}-dim vector\n")


def get_embedding(text: str) -> Optional[list]:
    """
    Tokenize text, run through CodeBERT, return CLS token vector (768-dim).
    Input is truncated to 512 tokens (CodeBERT's max).
    """
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token → (1, 768) → flatten to list
        cls_vector = outputs.last_hidden_state[:, 0, :]
        return cls_vector.squeeze().cpu().tolist()

    except Exception as e:
        print(f"    ❌ Embedding error: {repr(e)}")
        return None


# ─────────────────────────────────────────────
# PINECONE SETUP
# ─────────────────────────────────────────────

def init_pinecone():
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in .env")

    pc       = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if CODE_INDEX_NAME not in existing:
        print(f"  Creating index: {CODE_INDEX_NAME} (dim={EMBEDDING_DIM})")
        pc.create_index(
            name=CODE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(CODE_INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(3)
        print(f"  ✅ Index created: {CODE_INDEX_NAME}")
    else:
        print(f"  ✅ Connected to index: {CODE_INDEX_NAME}")

    return pc.Index(CODE_INDEX_NAME)


# ─────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────

CAD_CATEGORIES = {
    "rotor":      ["rotor", "disc", "disk", "brake"],
    "gear":       ["gear", "pinion", "rack", "tooth"],
    "shaft":      ["shaft", "axle", "crankshaft", "camshaft"],
    "housing":    ["housing", "casing", "enclosure"],
    "bracket":    ["bracket", "mount", "flange", "plate"],
    "piston":     ["piston", "cylinder", "connecting"],
    "bearing":    ["bearing", "race", "ring", "bushing"],
    "turbine":    ["turbine", "compressor", "blade", "wheel"],
    "fastener":   ["bolt", "screw", "nut", "stud"],
    "suspension": ["suspension", "control", "wishbone"],
}

def extract_metadata(filepath: Path, code: str) -> dict:
    stem_lower = filepath.stem.lower()
    code_lower = code.lower()

    category = "generic"
    for cat, keywords in CAD_CATEGORIES.items():
        if any(kw in stem_lower or kw in code_lower[:800] for kw in keywords):
            category = cat
            break

    line_count = code.count("\n")
    complexity = "simple"
    if line_count > 150:
        complexity = "complex"
    elif line_count > 60:
        complexity = "medium"

    return {
        "source_file": filepath.name,
        "category":    category,
        "complexity":  complexity,
        "line_count":  line_count,
        "code":        code,        # full code stored — used by LLM at query time
    }


# ─────────────────────────────────────────────
# BUILD EMBEDDING TEXT
# ─────────────────────────────────────────────

def build_embedding_text(stem: str, metadata: dict, code: str) -> str:
    name_readable = stem.replace("_", " ").replace("-", " ")

    ops = []
    if "revolve("  in code: ops.append("revolve")
    if "extrude("  in code: ops.append("extrude")
    if "sweep("    in code: ops.append("sweep")
    if ".fuse("    in code: ops.append("boolean union")
    if ".cut("     in code: ops.append("boolean cut")
    if ".fillet("  in code: ops.append("fillet")

    description = (
        f"Replicad CAD code example: {name_readable}. "
        f"Category: {metadata['category']}. "
        f"Operations: {', '.join(ops) if ops else 'basic'}. "
        f"Complexity: {metadata['complexity']}.\n\n"
    )

    # Cap at 2000 chars — CodeBERT truncates at 512 tokens anyway
    return description + code[:2000]


# ─────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────

def ingest(code_dir: str, dry_run: bool = False):
    print("\n" + "=" * 60)
    print("  CAD CODE INGESTION  (Microsoft CodeBERT)")
    print("=" * 60)

    code_path = Path(code_dir)
    if not code_path.exists():
        raise FileNotFoundError(f"Directory not found: {code_dir}")

    md_files = sorted(code_path.rglob("*.md"))
    if not md_files:
        print(f"  ⚠️  No .md files found in {code_dir}")
        return

    print(f"\n  Found {len(md_files)} .md files in: {code_dir}")

    init_codebert()
    index = init_pinecone() if not dry_run else None

    vectors       = []
    success_count = 0
    skip_count    = 0
    error_count   = 0

    for i, filepath in enumerate(md_files, 1):
        print(f"\n  [{i}/{len(md_files)}] {filepath.name}")

        try:
            code = filepath.read_text(encoding="utf-8").strip()
            if not code:
                print("    ⚠️  Empty — skipping")
                skip_count += 1
                continue

            metadata   = extract_metadata(filepath, code)
            embed_text = build_embedding_text(filepath.stem, metadata, code)

            print(f"    category={metadata['category']}  "
                  f"complexity={metadata['complexity']}  "
                  f"lines={metadata['line_count']}  "
                  f"embed_chars={len(embed_text)}")

            if dry_run:
                print(f"    [DRY RUN] skipping upload")
                success_count += 1
                continue

            embedding = get_embedding(embed_text)
            if embedding is None:
                print("    ❌ Embedding failed — skipping")
                error_count += 1
                continue

            print(f"    ✅ Embedded ({len(embedding)} dims)")

            vector_id = hashlib.md5(str(filepath).encode()).hexdigest()
            vectors.append({
                "id":       vector_id,
                "values":   embedding,
                "metadata": metadata,
            })
            success_count += 1

            if len(vectors) >= UPSERT_BATCH_SIZE:
                index.upsert(vectors=vectors)
                print(f"    📤 Upserted batch of {len(vectors)}")
                vectors = []

        except Exception as e:
            print(f"    ❌ Unexpected error: {repr(e)}")
            error_count += 1

    # Final partial batch
    if vectors and not dry_run:
        index.upsert(vectors=vectors)
        print(f"\n  📤 Final upsert: {len(vectors)} vectors")

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  ✅ Success : {success_count}")
    print(f"  ⏭️  Skipped : {skip_count}")
    print(f"  ❌ Errors  : {error_count}")

    if not dry_run and index:
        stats = index.describe_index_stats()
        print(f"  📦 Total vectors in Pinecone: {stats['total_vector_count']}")
    print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     default=DEFAULT_CODE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        ingest(args.dir, args.dry_run)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {repr(e)}")
        sys.exit(1)
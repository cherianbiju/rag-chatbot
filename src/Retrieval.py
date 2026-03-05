import os
import re
import time
import torch
import numpy as np
import sys
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from llama_index.llms.google_genai import GoogleGenAI
from typing import Optional

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY")
CODE_INDEX_NAME      = os.getenv("CODE_INDEX_NAME", "cad-code-examples")

LLM_MODEL            = "models/gemini-3-flash-preview"
FIX_LLM_MODEL        = "models/gemini-3-flash-preview"
QUERY_EXPAND_MODEL   = "models/gemini-3-flash-preview"

EMBEDDING_MODEL      = "microsoft/codebert-base"
SIMILARITY_THRESHOLD = 0.75
MAX_TOKENS           = 512
MAX_FIX_ATTEMPTS     = 3

BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH   = os.path.join(BASE_DIR, "replicad_system_prompt.js")
API_RULES_PATH       = os.path.join(BASE_DIR, "readme.adoc")


# ─────────────────────────────────────────────
# LOAD API RULES
# ─────────────────────────────────────────────

def load_api_rules() -> str:
    if not os.path.exists(API_RULES_PATH):
        print(f"  ⚠️  API rules not found at: {API_RULES_PATH}")
        return ""
    with open(API_RULES_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────
# CODEBERT EMBEDDING
# ─────────────────────────────────────────────

tokenizer = None
model     = None

def init_codebert():
    global tokenizer, model
    print(f"  Loading CodeBERT ({EMBEDDING_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    print(f"  ✅ CodeBERT ready")


def get_embedding(text: str) -> Optional[list]:
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        cls_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        norm       = np.linalg.norm(cls_vector)
        cls_vector = cls_vector / norm if norm > 0 else cls_vector
        return cls_vector.tolist()

    except Exception as e:
        print(f"  ❌ Embedding error: {e}")
        return None


# ─────────────────────────────────────────────
# GEMINI QUERY EXPANSION
# ─────────────────────────────────────────────

def expand_query_with_gemini(raw_query: str, expand_llm) -> str:
    expansion_prompt = f"""You are a CAD expert helping retrieve the best matching Replicad code example from a vector database.

The database stores JavaScript Replicad CAD code. The embeddings were generated from code text using CodeBERT.

Your job: rewrite the user's part description into a rich technical description that describes:
1. The part name and type
2. Key geometric features (e.g. revolve, extrude, sweep, fillet, boolean cut/union)
3. Likely code structure and Replicad API calls that would appear in the code
4. Category (gear, shaft, housing, bearing, bracket, rotor, piston, turbine, fastener, suspension)
5. Approximate complexity (simple / medium / complex)

Keep it concise (3-5 sentences). Write it as a technical code description, NOT as instructions to the LLM.
Do NOT write code. Do NOT explain. Just output the expanded description.

User query: {raw_query}

Expanded technical description:"""

    print(f"  🧠 Gemini expanding query for better retrieval...")
    try:
        response = expand_llm.complete(expansion_prompt)
        expanded = response.text.strip()
        print(f"  📝 Expanded: {expanded[:120]}...")
        return expanded
    except Exception as e:
        print(f"  ⚠️  Query expansion failed ({e}) — using raw query")
        return raw_query


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_example(query: str, index, expand_llm) -> Optional[dict]:
    print(f"\n🔍 Searching for similar example...")

    expanded_query = expand_query_with_gemini(query, expand_llm)

    embedding = get_embedding(expanded_query)
    if embedding is None:
        print("  ❌ Could not embed expanded query — trying raw query")
        embedding = get_embedding(query)
        if embedding is None:
            print("  ❌ Could not embed raw query either")
            return None

    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
    )

    matches = results.get("matches", [])
    if not matches:
        print("  ❌ No matches found")
        return None

    all_words = set(
        [w for w in query.lower().split()          if len(w) > 3] +
        [w for w in expanded_query.lower().split() if len(w) > 3]
    )
    for match in matches:
        filename  = match["metadata"].get("source_file", "").lower()
        word_hits = sum(1 for w in all_words if w in filename)
        if word_hits > 0:
            match["score"] += 1.5 * word_hits

    matches.sort(key=lambda x: x["score"], reverse=True)

    print(f"  📊 Top matches:")
    for i, m in enumerate(matches[:3], 1):
        fname = m["metadata"].get("source_file", "unknown")
        print(f"     {i}. {fname}  (score: {m['score']:.3f})")

    best     = matches[0]
    score    = best["score"]
    filename = best["metadata"].get("source_file", "unknown")

    if score >= SIMILARITY_THRESHOLD:
        code = best["metadata"].get("code", "")
        print(f"  ✅ Best match: {filename} (score: {score:.3f})")
        return {"filename": filename, "score": score, "code": code}
    else:
        print(f"  ⚠️  Best match below threshold: {filename} ({score:.3f}) — falling back to API rules")
        return None


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

def build_prompt(query: str, api_rules: str, example: Optional[dict]) -> str:
    if example:
        return f"""You are generating Replicad 3D CAD code for a mechanical part.

A similar working example has been retrieved. Adapt it for the requested part.
Keep the same code structure and Replicad API patterns.
Only change what is necessary to match the new part description.

# RETRIEVED EXAMPLE (adapt this)
File: {example['filename']} (score: {example['score']:.3f})
{example['code']}

# PART TO GENERATE
{query}

# STRICT RULES
- Return ONLY the complete javascript code
- No explanation, no markdown fences
- Follow the same API patterns shown in the example
- main() must return the final shape

Generate now."""

    else:
        return f"""You are generating Replicad 3D CAD code for a mechanical part.

No similar example was found in the database.
Generate from scratch using the API rules below.

# API RULES
{api_rules}

# PART TO GENERATE
{query}

# STRICT RULES
- Return ONLY the complete javascript code
- No explanation, no markdown fences
- main() must return the final shape
- Every shape reused in .fuse()/.cut()/.intersect() must call .clone() first
- lineTo() always takes [x, y] array — never two separate args
- Sketcher is consumed after .extrude()/.revolve() — never reuse it
- draw() must call .sketchOnPlane() before .extrude()
- No export default, no import, no module.exports

Generate now."""


def build_fix_prompt(error_message: str, broken_code: str) -> str:
    return f"""Fix this Replicad JavaScript code. It produced the error below.

Common causes:
- lineTo() called with two args instead of [x, y] array
- Sketcher reused after .extrude() or .revolve()
- draw() used without .sketchOnPlane() before .extrude()
- Shape reused in .fuse()/.cut() without .clone()
- Boolean op on non-overlapping geometry
- export default / module.exports / import — not allowed

# ERROR
{error_message}

# BROKEN CODE
{broken_code}

Return ONLY the corrected javascript code. No explanation. No markdown fences."""


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

def call_llm(prompt: str, llm, max_retries: int = 5) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                time.sleep(5)
            start    = time.time()
            response = llm.complete(prompt)
            elapsed  = time.time() - start
            print(f"  LLM completed in {elapsed:.1f}s")
            return response.text
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_retries - 1:
                print(f"  Rate limited — waiting 60s... ({attempt+1}/{max_retries})")
                time.sleep(60)
                continue
            if ("503" in err or "504" in err or "UNAVAILABLE" in err) and attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  Server unavailable — waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"\n❌ LLM Error: {e}")
            return None
    return None


# ─────────────────────────────────────────────
# PROMPT HELPER  (terminal mode only)
# ─────────────────────────────────────────────

def prompt_input(message: str) -> str:
    sys.stdout.write(message)
    sys.stdout.flush()
    try:
        return sys.stdin.readline().rstrip("\n")
    except EOFError:
        return ""


# ─────────────────────────────────────────────
# ERROR FIX LOOP  (terminal mode only)
# ─────────────────────────────────────────────

def error_fix_loop(code: str, api_rules: str, fix_llm) -> str:
    current_code = code
    attempt = 0

    while attempt < MAX_FIX_ATTEMPTS:

        print("\n" + "-" * 70)
        print("📋 Paste the code above into Replicad studio and run it.")
        print("-" * 70)

        answer = prompt_input("\nDid the code produce an error? (y/n): ").strip().lower()

        if answer != "y":
            print("\n✅ Code accepted.")
            return current_code

        print("\nPaste the error message below.")
        print("Press Enter on a BLANK line when done:\n")
        sys.stdout.flush()

        error_lines = []
        while True:
            line = prompt_input("")
            if line == "":
                break
            error_lines.append(line)

        error_message = "\n".join(error_lines).strip()

        if not error_message:
            print("  No error message entered — accepting code as-is.")
            return current_code

        attempt += 1
        print(f"\n🔧 Fixing (attempt {attempt}/{MAX_FIX_ATTEMPTS})...")

        fix_start  = time.time()
        fixed_code = call_llm(
            build_fix_prompt(error_message, current_code),
            fix_llm
        )
        print(f"  ⏱️  Fix took {time.time() - fix_start:.1f}s")

        if not fixed_code:
            print("  ❌ Fix LLM call failed — returning last version.")
            return current_code

        current_code = fixed_code

        print("\n" + "=" * 70)
        print(f"🔧 FIXED CODE (attempt {attempt})")
        print("=" * 70)
        print(current_code)

    print(f"\n⚠️  Max fix attempts ({MAX_FIX_ATTEMPTS}) reached.")
    return current_code


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def setup():
    print("Setting up RAG system...")

    system_prompt = ""
    if os.path.exists(SYSTEM_PROMPT_PATH):
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    else:
        print(f"  ⚠️  System prompt not found at {SYSTEM_PROMPT_PATH} — continuing without it")

    llm = GoogleGenAI(
        model=LLM_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.1,
        request_timeout=300,
        max_retries=3,
        system_prompt=system_prompt,
    )

    fix_llm = GoogleGenAI(
        model=FIX_LLM_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.1,
        request_timeout=300,
        max_retries=3,
        system_prompt=system_prompt,
    )

    expand_llm = GoogleGenAI(
        model=QUERY_EXPAND_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.0,
        request_timeout=60,
        max_retries=3,
    )

    init_codebert()

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(CODE_INDEX_NAME)

    print("✅ RAG ready\n")
    return llm, fix_llm, expand_llm, index


# ─────────────────────────────────────────────
# MAIN QUERY FLOW
# fix_loop=True  → terminal mode (interactive y/n error loop)
# fix_loop=False → Streamlit mode (returns code immediately)
# ─────────────────────────────────────────────

def run_query(user_input: str, llm, fix_llm, expand_llm, index, fix_loop: bool = True):
    print("\n" + "=" * 70)
    print(f"QUERY: {user_input}")
    print("=" * 70)

    api_rules = load_api_rules()
    if not api_rules:
        print("⚠️  API rules empty — LLM will use own knowledge only")

    example = retrieve_example(user_input, index, expand_llm)
    prompt  = build_prompt(user_input, api_rules, example)

    print(f"\n{'=' * 70}")
    if example:
        print(f"🤖 GENERATING — adapting example: {example['filename']}")
    else:
        print(f"🤖 GENERATING — from scratch using API rules")
    print(f"{'=' * 70}")

    final_code = call_llm(prompt, llm)
    if not final_code:
        print("\n❌ Generation failed.")
        return None

    print("\n" + "=" * 70)
    print("✅ GENERATED CODE")
    print("=" * 70)
    print(final_code)

    # Skip interactive terminal loop when called from Streamlit
    if fix_loop:
        final_code = error_fix_loop(final_code, api_rules, fix_llm)

    return final_code


# ─────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────

try:
    llm, fix_llm, expand_llm, pinecone_index = setup()
except Exception as e:
    print(f"❌ Setup failed: {e}")
    llm            = None
    fix_llm        = None
    expand_llm     = None
    pinecone_index = None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not llm:
        print("System not initialized. Exiting.")
        sys.exit(1)

    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]), llm, fix_llm, expand_llm, pinecone_index)
    else:
        print("\nType 'exit' to quit\n")
        while True:
            try:
                user_input = prompt_input("Question: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    print("\nEnding the querying!")
                    break
                run_query(user_input, llm, fix_llm, expand_llm, pinecone_index)
            except KeyboardInterrupt:
                print("\nEnding the querying!")
                break
import os
import re
import time
import sys
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
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

LLM_MODEL            = "models/gemini-3.1-pro-preview"
EMBEDDING_MODEL      = "gemini-embedding-001"        # must match ingest.py

SIMILARITY_THRESHOLD = 0.75
MAX_FIX_ATTEMPTS     = 1

BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH   = os.path.join(BASE_DIR, "replicad_system_prompt.js")
API_RULES_PATH       = os.path.join(BASE_DIR, "readme.adoc")

gemini_embed_client = None


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
# GEMINI EMBEDDING  (must match ingest.py)
# ─────────────────────────────────────────────

def init_gemini_embedding():
    global gemini_embed_client
    gemini_embed_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )
    print(f"  ✅ Gemini embedding client ready ({EMBEDDING_MODEL})")


def get_embedding(text: str, retries: int = 3) -> Optional[list]:
    global gemini_embed_client
    for attempt in range(retries):
        try:
            response = gemini_embed_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",  # QUERY for search, DOCUMENT for ingest
                ),
            )
            # 3072 dims — already normalized by Gemini
            return list(response.embeddings[0].values)
        except Exception as e:
            err = repr(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < retries - 1:
                print(f"    Rate limited — waiting 30s...")
                time.sleep(30)
                continue
            if ("503" in err or "UNAVAILABLE" in err) and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"    Server unavailable — waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  ❌ Embedding error: {err}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                return None
    return None


# ─────────────────────────────────────────────
# QUERY EXPANSION  (Gemini)
# ─────────────────────────────────────────────

def expand_query(raw_query: str, llm) -> str:
    expansion_prompt = f"""You are a CAD expert helping retrieve Replicad JavaScript CAD code from a vector database.

Each stored document was embedded with this prefix style:
"Replicad CAD code example: <part name>. Category: <category>. Operations: <op1, op2>. Complexity: <level>."
followed by the actual JavaScript code.

Your job: rewrite the user query to match this style, then add likely function/variable names from the code.

Categories: gear, shaft, housing, bearing, bracket, rotor, piston, turbine, fastener, suspension, generic
Operations: revolve, extrude, sweep, boolean union, boolean cut, fillet

Output only the rewritten description. No explanation.

User query: {raw_query}

Rewritten:"""

    print(f"  🧠 Expanding query...")
    try:
        response = llm.complete(expansion_prompt)
        expanded = response.text.strip()
        print(f"  📝 Expanded: {expanded[:120]}...")
        return expanded
    except Exception as e:
        print(f"  ⚠️  Query expansion failed ({e}) — using raw query")
        return raw_query


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

def retrieve_example(query: str, index, llm) -> Optional[dict]:
    print(f"\n🔍 Searching for similar example...")

    expanded_query = expand_query(query, llm)

    embedding = get_embedding(expanded_query)
    if embedding is None:
        print("  ❌ Could not embed expanded query — trying raw query")
        embedding = get_embedding(query)
        if embedding is None:
            print("  ❌ Embedding failed entirely")
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

    # Re-rank — boost filename word matches
    all_words = set(
        [w for w in query.lower().split()          if len(w) > 3] +
        [w for w in expanded_query.lower().split() if len(w) > 3]
    )
    for match in matches:
        filename  = match["metadata"].get("file", "").lower()
        word_hits = sum(1 for w in all_words if w in filename)
        if word_hits > 0:
            match["score"] += 1.5 * word_hits

    matches.sort(key=lambda x: x["score"], reverse=True)

    print(f"  📊 Top matches:")
    for i, m in enumerate(matches[:3], 1):
        fname = m["metadata"].get("file", "unknown")
        print(f"     {i}. {fname}  (score: {m['score']:.3f})")

    best     = matches[0]
    score    = best["score"]
    filename = best["metadata"].get("file", "unknown")

    if score >= SIMILARITY_THRESHOLD:
        # ── Strict filename match check ──────────────────────────────
        # Extract the core part name from the query (remove generic words
        # like "create", "generate", "make", "a", "the", "with" etc.)
        STOP_WORDS = {
            "create","generate","make","build","design","model","draw",
            "give","show","produce","write","code","replicad","cad",
            "with","and","for","the","that","has","from","using",
            "a","an","of","in","on","at","to","is","are","gear" # "gear" alone is too generic
        }
        query_words = [
            w for w in query.lower().split()
            if len(w) > 3 and w not in STOP_WORDS
        ]
        fname_clean = filename.replace("_", " ").replace("-", " ").replace(".md", "").lower()

        # All meaningful query words must appear in the filename
        strict_match = all(w in fname_clean for w in query_words) if query_words else False

        if not strict_match:
            print(f"  ⚠️  Filename '{filename}' does not strictly match query words {query_words}")
            print(f"       → Rejecting match, falling back to API rules + geometry hints")
            return None

        code = best["metadata"].get("code", "")
        print(f"  ✅ Strict match confirmed: {filename} (score: {score:.3f})")
        return {"filename": filename, "score": score, "code": code}
    else:
        print(f"  ⚠️  Below threshold: {filename} ({score:.3f}) — falling back to API rules")
        return None


# ─────────────────────────────────────────────
# GEOMETRY HINTS
# Injected into the prompt when no DB match is
# found — guides the LLM to generate the correct
# geometry instead of falling back to a simpler shape
# ─────────────────────────────────────────────

GEOMETRY_HINTS = {
    "worm gear": """A worm gear is NOT a disc gear. It has TWO completely different parts:

PART 1 - THE WORM (looks like a bolt/screw):
- Start with a cylinder (the shaft), e.g. radius=10, length=60
- Cut a helical groove around it using a swept profile to create the thread
- The result must look like a metal screw/bolt, NOT a gear disc

PART 2 - THE WORM WHEEL (looks like a spur gear with a groove):
- Start with a disc with involute teeth around the perimeter
- Add a concave groove around the middle of the teeth face to cradle the worm
- Has a central bore for a shaft

Return BOTH parts positioned next to each other (worm horizontal, wheel vertical, meshing together).

CRITICAL: The worm MUST be cylindrical like a screw. If you generate a disc shape for the worm, you are WRONG.""",

    "bevel gear": """A bevel gear has teeth cut on a conical surface — teeth taper toward the apex.
Generate a truncated cone, then cut tapered tooth profiles around its outer surface using boolean cuts.
Key parameters: pitch angle, number of teeth, module, face width.
Do NOT generate a spur gear. The blank must be conical, not cylindrical.""",

    "rack": """A gear rack is a flat bar with straight teeth on one face — it meshes with a pinion gear.
Generate a rectangular bar, then cut evenly-spaced triangular/involute tooth profiles along one face.
Key parameters: module, number of teeth, tooth height, rack length, width, thickness.""",

    "sprocket": """A sprocket has evenly spaced teeth designed to engage roller chain links.
Generate a disc with D-shaped tooth profiles cut around the perimeter. Add a central bore.
Key parameters: number of teeth, pitch, roller diameter, bore diameter.""",

    "cycloidal": """A cycloidal gear uses cycloidal curves for tooth profiles instead of involute curves.
Generate the tooth profile using parametric cycloidal math and extrude.
Key parameters: number of teeth, disc radius, roller radius, eccentricity.""",

    "cam": """A cam converts rotational motion to linear motion via a non-circular profile.
Generate by extruding a non-circular 2D profile (e.g. egg-shaped, heart-shaped, or eccentric circle).
Include a central shaft bore. Key parameters: base circle radius, lift amount, cam width.""",

    "impeller": """An impeller has curved blades radiating from a central hub for fluid movement.
Generate a disc hub, then sweep curved blade profiles around it at an angle.
Key parameters: number of blades, inner radius, outer radius, blade angle, blade thickness.""",
}

def _geometry_hint(query: str) -> str:
    """Return geometry-specific guidance based on keywords in the query."""
    q = query.lower()
    for keyword, hint in GEOMETRY_HINTS.items():
        if keyword in q:
            return hint
    return "Generate the exact geometry described. Do not simplify or substitute with a different part type."


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

def build_prompt(query: str, api_rules: str, example: Optional[dict], system_prompt: str = "") -> str:
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
        geometry_hint = _geometry_hint(query)

        return f"""You are generating Replicad 3D CAD code for a mechanical part.

No similar example was found in the database.
You MUST generate the correct geometry from scratch — do NOT fall back to a simpler shape.
For example, a worm gear is NOT a spur gear. A bevel gear is NOT a spur gear. Generate the exact requested geometry.

# SYSTEM KNOWLEDGE (Replicad API rules and patterns)
{system_prompt}

# API RULES
{api_rules}

# GEOMETRY GUIDANCE FOR THIS PART
{geometry_hint}

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
# LLM CALL  (Gemini)
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

def error_fix_loop(code: str, api_rules: str, llm) -> str:
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
        fixed_code = call_llm(build_fix_prompt(error_message, current_code), llm)
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

    init_gemini_embedding()

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(CODE_INDEX_NAME)

    print("✅ RAG ready\n")
    return llm, index, system_prompt


# ─────────────────────────────────────────────
# MAIN QUERY FLOW
# fix_loop=True  → terminal mode
# fix_loop=False → Streamlit mode
# ─────────────────────────────────────────────

def run_query(user_input: str, llm, index, system_prompt: str = "", fix_loop: bool = True):
    print("\n" + "=" * 70)
    print(f"QUERY: {user_input}")
    print("=" * 70)

    api_rules = load_api_rules()
    if not api_rules:
        print("⚠️  API rules empty — LLM will use own knowledge only")

    # If similar found → use file to generate, else → use api_rules + LLM
    example = retrieve_example(user_input, index, llm)
    prompt  = build_prompt(user_input, api_rules, example, system_prompt)

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

    if fix_loop:
        final_code = error_fix_loop(final_code, api_rules, llm)

    return final_code


# ─────────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────────

try:
    llm, pinecone_index, system_prompt = setup()
except Exception as e:
    print(f"❌ Setup failed: {e}")
    llm            = None
    pinecone_index = None
    system_prompt  = ""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not llm:
        print("System not initialized. Exiting.")
        sys.exit(1)

    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]), llm, pinecone_index)
    else:
        print("\nType 'exit' to quit\n")
        while True:
            try:
                user_input = prompt_input("Question: ").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    print("\nEnding the querying!")
                    break
                run_query(user_input, llm, pinecone_index, system_prompt=system_prompt)
            except KeyboardInterrupt:
                print("\nEnding the querying!")
                break
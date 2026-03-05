import streamlit as st
import time
import re
from Retrieval import setup, run_query, build_fix_prompt, call_llm

st.set_page_config(
    page_title="Replicad RAG",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Replicad RAG System")


# ─────────────────────────────────────────────
# Initialize RAG — unpack the tuple from setup()
# ─────────────────────────────────────────────
@st.cache_resource
def initialize_rag():
    llm, fix_llm, expand_llm, index = setup()
    return llm, fix_llm, expand_llm, index

llm, fix_llm, expand_llm, pinecone_index = initialize_rag()


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🛠️ Generate Code", "🐛 Fix My Error"])


# ─────────────────────────────────────────────
# TAB 1 — Generate Code
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Generate Replicad CAD Code")
    user_query = st.text_input("Enter your CAD question:", key="gen_query")

    if st.button("Generate Code") and user_query:
        with st.spinner("Retrieving & Generating..."):
            try:
                start = time.time()

                generated_code = run_query(
                    user_query, llm, fix_llm, expand_llm, pinecone_index,
                    fix_loop=False   # ← skip terminal interaction in Streamlit
                )
                elapsed = time.time() - start

                if generated_code:
                    st.success(f"Completed in {elapsed:.2f}s")

                    generated_code = re.sub(r"```[a-zA-Z]*", "", generated_code)
                    generated_code = generated_code.replace("```", "").strip()

                    st.subheader("🤖 Generated Code")
                    st.code(generated_code, language="javascript")
                else:
                    st.error("Generation failed — check terminal logs for details.")

            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# TAB 2 — Fix My Error
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Paste your error and get a fix")

    col1, col2 = st.columns(2)

    with col1:
        error_msg = st.text_area(
            "❌ Paste your error message:",
            height=180,
            placeholder="e.g. TypeError: sketchCircle is not a function..."
        )

    with col2:
        broken_code = st.text_area(
            "📋 Paste your broken code:",
            height=180,
            placeholder="Paste the Replicad JS code that caused the error..."
        )

    if st.button("Fix Error"):
        if not error_msg:
            st.warning("Please paste your error message.")
        elif not broken_code:
            st.warning("Please paste your broken code too — it's needed to generate a fix.")
        else:
            with st.spinner("Analyzing error & generating fix..."):
                try:
                    start = time.time()

                    fix_prompt = build_fix_prompt(error_msg, broken_code)
                    fixed_code = call_llm(fix_prompt, fix_llm)
                    elapsed    = time.time() - start

                    if fixed_code:
                        st.success(f"Completed in {elapsed:.2f}s")

                        fixed_code = re.sub(r"```[a-zA-Z]*", "", fixed_code)
                        fixed_code = fixed_code.replace("```", "").strip()

                        st.subheader("✅ Fixed Code")
                        st.code(fixed_code, language="javascript")
                    else:
                        st.error("Fix failed — check terminal logs for details.")

                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
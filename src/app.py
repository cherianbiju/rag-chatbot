import streamlit as st
import time
import re

from Retrieval import setup

st.set_page_config(
    page_title="Replicad RAG",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Testing RAG System")
st.markdown("Generate parametric Replicad CAD code using RAG")

# Initialize RAG system once
@st.cache_resource
def initialize_rag():
    return setup()

query_engine = initialize_rag()

# User input
user_query = st.text_input("Enter your CAD question:")

if st.button("Generate Code") and user_query:

    with st.spinner("Retrieving & Generating..."):
        try:
            start = time.time()
            response = query_engine.query(user_query)
            elapsed = time.time() - start

            st.success(f"Completed in {elapsed:.2f}s")

            # Clean markdown code blocks if present
            generated_code = response.response
            generated_code = re.sub(r"```[a-zA-Z]*", "", generated_code)
            generated_code = generated_code.replace("```", "").strip()

            # Display clean code
            st.subheader("🤖 Generated Code")
            st.code(generated_code, language="javascript")

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("RAG-based Parametric CAD Code Generation System")

import streamlit as st
from rag_pipeline import retrieve_relevant_chunks
from ollama_client import ask_ollama

# This must be the first Streamlit command
st.set_page_config(layout="wide")

@st.cache_resource
def load_resources():
    # This function is just a placeholder for caching
    pass

# Add a query cache
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def get_answer(query):
    # Retrieve relevant chunks from the PDF
    retrieved_chunks, sources = retrieve_relevant_chunks(query, top_k=2)
    
    # Call Ollama LLM with the retrieved context
    answer = ask_ollama(query, retrieved_chunks, temperature=0.1, max_tokens=512)
    
    return answer, sources

st.title("PCSOFT IEV Help Assistant")
query = st.text_input("Ask a question about PCSOFT IEV reports:")

if query:
    with st.spinner("Processing your question..."):
        # Use the cached function
        answer, sources = get_answer(query)
    
    st.write("**Answer:**")
    st.write(answer)
    st.write("---")
    st.write("**Source Snippets:**")
    for i, chunk in enumerate(sources):
        st.write(f"**Snippet {i+1}:**")
        st.write(chunk)
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Correct endpoint for Ollama

SYSTEM_PROMPT = (
    "You are a help assistant for PCSOFT IEV. "
    "Only answer using the provided documentation context. "
    "If the answer is not found, reply: 'This information is not available in the current documentation.'"
)

def ask_ollama(query, context_chunks, model="llama3"):
    # Combine context chunks into a single context string
    context = "\n\n".join(context_chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nUser question: {query}\nAnswer:"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "context": context  # Added context field for better results
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return "Sorry, I couldn't process your request right now."
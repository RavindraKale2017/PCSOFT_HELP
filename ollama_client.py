import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
session = requests.Session()  # Create a persistent session

def ask_ollama(query, context_chunks, model="llama3"):
    # Combine context chunks into a single context string
    context = "\n\n".join(context_chunks)
    prompt = f"You are a help assistant for PCSOFT IEV. Only answer using the provided documentation context.\n\nContext:\n{context}\n\nUser question: {query}\nAnswer:"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Disable streaming for faster responses
    }
    
    try:
        response = session.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return "Sorry, I couldn't process your request right now."
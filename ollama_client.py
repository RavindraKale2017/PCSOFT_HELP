import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
session = requests.Session()  # Create a persistent session

def ask_ollama(query, context_chunks, model="llama3", temperature=0.3, max_tokens=512):
    context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
    
    prompt = f"""You are a technical support assistant for PCSOFT IEV software. 
Answer the user's question using ONLY the provided context. If the answer isn't in the context, say "I couldn't find that information in the documentation."

Context:
{context}

Question: {query}

Answer guidelines:
1. Be concise and technical
2. Use bullet points if listing steps
3. Include relevant parameter names and values
4. Reference the context number if applicable

Answer:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = session.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return "Sorry, I couldn't process your request right now."
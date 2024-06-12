# RAG-with-GROQ-pgvector

### Vector Encoding and Storage
- SentenceTransformer: Encodes text into 384-dimensional vectors.
- Supabase: Stores vectors using the vecs library.


### Retrieval and Generation
- Retrieval: Finds relevant vectors based on a query.
- Generation: Uses the Groq API to generate answers from the context and chat history.

### FastAPI Server
- Endpoint: Provides an API to interact with the RAG system.
- Chat History: Maintains history for coherent responses.

### Usage
*Gradio*  <br>
Or <br> 
*FastAPI Endpoint*
```
{
  "query": "Your question here",
  "return_context": true
}
```

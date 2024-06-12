from groq import Groq
import vecs
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os
 
load_dotenv()


DB_CONNECTION = os.getenv('SupaBaseConnection')
client = Groq(api_key=os.getenv('GroqApiKey'),)

# Connecting to Supabase
vx = vecs.create_client(DB_CONNECTION)
vector = vx.get_or_create_collection(name="vectors_embeddings", dimension=384)

# Creating out Encoder.
model = SentenceTransformer('thenlper/gte-small')


def ask_groq(query, chat_history, return_context=False):
    print("----Embedding Query.")
    query_vector = model.encode([query])

    print("----Searching Similar Vectors in DB")
    # Finding similar vectors using euclidean distance
    result = vector.query(
            data=query_vector[0],
            limit=3, # Selecting 3 as told in assignment doc.
            filters={},
            measure="l2_distance",
            include_value=True,
            include_metadata=True,
        )

    context = """"""
    for vec_id, euclidean_distance, metadata in result:
        print("euclidean_distance: "+str(euclidean_distance))
        context += metadata['text']+"\n\n"
    
    history = """"""
    for question, answer in chat_history:
        history+=f"user: {question}\nsystem: {answer}\n\n"

    print("\n----Asking Groq with Context")
    chat_completion = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content": f"""
                        Instructions:
                        You are an intelligent agent trained to answer questions based on a given context. Your task is to carefully read the provided context and use it to answer the given question as accurately as possible.
                        
                        Do Not Answer any thing which is not in Context, Even if yoou know the answer.

                        If the context contains enough information to answer the question:
                        1. Read the context thoroughly and understand its content.
                        2. Analyze the question and identify the key information needed to answer it.
                        3. Search the context for relevant information to formulate your answer.
                        4. Provide a clear and concise answer based on the information found in the context.

                        If the context does not contain enough information to answer the question:
                        1. Respond with: "I'm sorry, the provided context does not contain enough information to answer the question '[restate the original question]'. I cannot provide a complete answer based on the given context."
                        2. Do not attempt to speculate or make up information not found in the context.

                        In both cases, you must:
                        - Provide your response in a polite and helpful manner.
                        - Avoid making assumptions or using external knowledge not contained in the given context.
                        - If the context is irrelevant or unrelated to the question, respond with: "The provided context is not relevant to answering the question '[restate the original question]'."

                        Context: 
                        ----- starts ----
                        {context}
                        ---- ends ----

                        Chat History:
                        {history}
                        """
            },
            {
                "role": "user",
                "content": query,
            }
        ],
        model="mixtral-8x7b-32768",
    )  

    if return_context:
        return chat_completion.choices[0].message.content, context
    else: 
        return chat_completion.choices[0].message.content

if __name__=="__main__":
    import gradio as gr
    gr.ChatInterface(ask_groq).launch()

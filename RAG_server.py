from fastapi import FastAPI
from pydantic import BaseModel
from GropLLM import ask_groq

app = FastAPI()

class UserInput(BaseModel):
    query: str
    return_context: bool = False

ChatHistory = []

@app.post("/")
async def ask_RAG(userinput: UserInput):
    global ChatHistory
    result = ask_groq(query=userinput.query, chat_history=ChatHistory, return_context=userinput.return_context)

    if userinput.return_context:
        result, context = result

    ChatHistory.append([userinput.query, result])
    # print(ChatHistory)

    response=  {'result': result}
    if userinput.return_context:
        response['context']= context
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6969)
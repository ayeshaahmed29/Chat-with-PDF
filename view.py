from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chat

app= FastAPI(
    title="chat with PDF API",
    description="Ask questions about a PDF through an API using langchain and RAG",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Chat with PDF API is running..."}

@app.post("/Ask")
async def ask_pdf(request: QuestionRequest):
    ragchain_instance= chat.rag_chain
    if ragchain_instance is None:
        raise HTTPException(status_code=503, detail="RAG not initialized properly")
    try:
        response= await ragchain_instance.ainvoke({"input": request.question})
        return {"Question":request.question, "Answer": response["answer"]}
    except:
        raise HTTPException(status_code=500, detail="Error in processing question...")
    
    
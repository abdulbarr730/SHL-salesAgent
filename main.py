import logging
from typing import List, Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from agent import process_chat

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/docs")

# Configure logging to monitor agent behavior
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0"
)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}

@app.post("/chat", status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Processes stateless conversation history and returns recommendations.
    """
    try:
        # Map Pydantic models to dict format for the agent
        history = [msg.model_dump() for msg in request.messages]
        logger.info(f"Chat request received. Turns: {len(history)}")
        
        response = process_chat(history)
        return response
        
    except Exception as e:
        logger.error(f"Agent processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to generate response."
        )
import logging
from typing import List, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from agent import process_chat

# 1. Configure logging to monitor agent behavior and debug deployment issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Single FastAPI initialization with custom metadata
app = FastAPI(
    title="SHL Assessment Recommender",
    description="A lightweight, high-performance RAG agent for SHL test solutions.",
    version="1.0.0"
)

# 3. Request/Response Validation Schemas using Pydantic
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

# 4. API Endpoints

@app.get("/")
async def root_redirect():
    """
    Automatically redirects root visitors to interactive Swagger documentation.
    Prevents 404/Not Found errors when opening the base Render URL.
    """
    return RedirectResponse(url="/docs")

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Standard health check endpoint for Render monitoring.
    """
    return {"status": "ok"}

@app.post("/chat", status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Processes stateless conversation history and returns recommendations.
    Accepts conversation trace arrays and forwards them to the agent pipeline.
    """
    try:
        # Map Pydantic models to a list of dicts for processing by the agent
        history = [msg.model_dump() for msg in request.messages]
        logger.info(f"Chat request received. Current conversation depth: {len(history)} turns.")
        
        # Process the trace array through the agent + local retrieval logic
        response = process_chat(history)
        return response
        
    except Exception as e:
        logger.error(f"Agent processing failed at endpoint layer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to generate a valid agent response."
        )
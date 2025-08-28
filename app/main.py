import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tempfile
import json
import asyncio
from app.parser import parse_pdf_with_images, parse_pdf_with_images_async
from app.generator import generate_testcases, generate_testcases_multimodal, generate_testcases_with_rag
from llama_index.llms.gemini import Gemini


load_dotenv() #loads the environment variables from the .env file

app = FastAPI() #instantiates the FastAPI app

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate-testcases")
async def upload_and_generate(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
        
        try:
            # Use async version for better performance
            parsed_content = await parse_pdf_with_images_async(tmp_path)
            
            # Generate test cases using RAG
            testcases = generate_testcases_with_rag(parsed_content)
            
            return JSONResponse(content={"testcases": testcases})
            
        finally:
            os.remove(tmp_path)

@app.get("/test-rag")
async def test_rag():
    """Test endpoint to debug RAG system."""
    try:
        from app.rag_setup import get_qa_index
        qa_index = get_qa_index()
        llm = Gemini(model_name="gemini-1.5-flash")
        query_engine = qa_index.as_query_engine(
            llm=llm,
            response_mode="compact",
            streaming=False
        )
        
        # Test query
        test_query = "What are the key testing patterns for API testing?"
        response = query_engine.query(test_query)
        
        # Extract source nodes for debugging
        source_nodes = []
        if response.source_nodes:
            for node in response.source_nodes:
                source_nodes.append({
                    "text": node.get_content(),
                    "score": node.get_score(),
                    "node_id": node.node_id
                })
        
        return JSONResponse(content={
            "rag_working": True,
            "test_query": test_query,
            "response": str(response),
            "retrieved_nodes": source_nodes
        })
    except Exception as e:
        return JSONResponse(content={
            "rag_working": False,
            "error": str(e),
            "error_type": str(type(e))
        }, status_code=500)

@app.post("/reset-rag")
async def reset_rag():
    """Reset RAG system to use new API key."""
    try:
        from app.rag_setup import reset_qa_index
        reset_qa_index()
        return JSONResponse(content={
            "message": "RAG cache cleared successfully. New requests will use current API key.",
            "status": "success"
        })
    except Exception as e:
        return JSONResponse(content={
            "message": f"Failed to reset RAG: {str(e)}",
            "status": "error"
        }, status_code=500)



   



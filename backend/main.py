from fastapi import FastAPI, HTTPException
from backend.api import routes

app = FastAPI(title = "Causal Inference Analysis Tool")
app.include_router(routes.router, prefix="/api", tags=["routes"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API..."}

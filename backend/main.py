from fastapi import FastAPI, HTTPException

app = FastAPI(title = "Causal Inference Analysis Tool")

@app.get("/")
async def root():
    return {"message": "Welcome to the Semantic Search API..."}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="GreenBit API",
    description="AI-Driven Dark Data Minimization Framework",
    version="1.0.0"
)

# Enable CORS for frontend communication
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*"  # Allow all for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test routes
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "GreenBit: AI-Driven Dark Data Minimization Framework",
        "version": "1.0.0",
        "status": "🟢 Backend is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "✅ GreenBit Backend is healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/info")
def get_info():
    """API information endpoint"""
    return {
        "name": "GreenBit Backend API",
        "version": "1.0.0",
        "description": "AI-driven framework for dark data minimization",
        "endpoints": {
            "root": "/",
            "health": "/api/health",
            "info": "/api/info",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }

# Optional: Keep this if you want to run with `python main.py`
if __name__ == "__main__":
    import uvicorn
    # Run without reload to avoid the warning
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

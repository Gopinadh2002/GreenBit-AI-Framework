"""
GreenBit Backend API
Main FastAPI application with ML pipeline integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import shutil
import tempfile
import traceback

# Add ml_pipeline to Python path so we can import from it
ml_pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline'))
if ml_pipeline_path not in sys.path:
    sys.path.insert(0, ml_pipeline_path)

print(f"📁 ML Pipeline path: {ml_pipeline_path}")
print(f"✅ Path exists: {os.path.exists(ml_pipeline_path)}")

try:
    from pipeline import GreenBitPipeline
    ML_PIPELINE_AVAILABLE = True
    print("✅ Successfully imported GreenBitPipeline")
except ImportError as e:
    print(f"❌ Error importing pipeline: {e}")
    ML_PIPELINE_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="GreenBit API",
    description="AI-Driven Dark Data Minimization Framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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

# Initialize ML Pipeline
print("\n" + "="*70)
print("INITIALIZING GREENBIT BACKEND")
print("="*70)

if ML_PIPELINE_AVAILABLE:
    try:
        pipeline = GreenBitPipeline()
        print("✅ ML Pipeline loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error initializing ML Pipeline: {e}")
        pipeline = None
        ML_PIPELINE_AVAILABLE = False
else:
    pipeline = None

# Storage for analysis results (in-memory for now)
analysis_results = {}

# ============================================================================
# DATA MODELS
# ============================================================================

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    backend_version: str
    ml_pipeline_available: bool
    environment: str

class AnalysisRequest(BaseModel):
    """Analysis request model"""
    analysis_id: str
    timestamp: str
    status: str

class AnalysisSummary(BaseModel):
    """Analysis summary model"""
    total_files: int
    duplicates_found: int
    similar_files: int
    co2_emissions_kg: float

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "GreenBit: AI-Driven Dark Data Minimization Framework",
        "version": "1.0.0",
        "status": "🟢 Backend is running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "results": "/api/results/{analysis_id}",
            "analyses": "/api/analyses",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/api/health", response_model=HealthCheckResponse)
def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="✅ GreenBit Backend is healthy",
        timestamp=datetime.now().isoformat(),
        backend_version="1.0.0",
        ml_pipeline_available=ML_PIPELINE_AVAILABLE and pipeline is not None,
        environment=os.getenv("ENVIRONMENT", "development")
    )

@app.get("/api/info")
def get_info():
    """API information endpoint"""
    return {
        "name": "GreenBit Backend API",
        "version": "1.0.0",
        "description": "AI-driven framework for dark data minimization",
        "ml_pipeline_status": "available" if ML_PIPELINE_AVAILABLE and pipeline else "not_available",
        "endpoints": {
            "POST /api/analyze": "Analyze uploaded files",
            "GET /api/results/{analysis_id}": "Get analysis results",
            "GET /api/analyses": "List all analyses",
            "GET /api/health": "Health check",
            "GET /docs": "Swagger UI documentation",
            "GET /redoc": "ReDoc documentation"
        }
    }

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/analyze")
async def analyze_files(files: list[UploadFile] = File(...)):
    """
    Analyze uploaded files for duplicates and dark data
    
    Args:
        files: List of files to analyze
        
    Returns:
        Analysis ID and initial results
    """
    
    if not ML_PIPELINE_AVAILABLE or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ML Pipeline is not available"
        )
    
    try:
        print(f"\n📁 Received {len(files)} files for analysis")
        
        # Create temporary directory for uploads
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir(exist_ok=True)
            
            # Save uploaded files
            saved_files = []
            total_size = 0
            
            for file in files:
                if file.filename:
                    file_path = upload_dir / file.filename
                    contents = await file.read()
                    total_size += len(contents)
                    
                    with open(file_path, "wb") as f:
                        f.write(contents)
                    saved_files.append(str(file_path))
            
            print(f"✅ Saved {len(saved_files)} files ({total_size / 1024 / 1024:.2f} MB)")
            
            # Run analysis
            print("🔍 Running analysis...")
            results = pipeline.analyze_directory(str(upload_dir), max_files=None)
            
            # Generate unique analysis ID
            analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_results[analysis_id] = results
            
            print(f"✅ Analysis complete: {analysis_id}")
            
            # Prepare response
            if "error" in results:
                raise HTTPException(status_code=400, detail=results["error"])
            
            return {
                "status": "✅ Analysis complete",
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "files_processed": len(saved_files),
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "duplicates_found": len(results.get("similarity_analysis", {}).get("similar_pairs", [])),
                "clusters_found": results.get("clustering_results", {}).get("total_clusters", 0),
                "co2_emissions_kg": results.get("carbon_emissions", {}).get("inference_emissions_kg", 0)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/api/results/{analysis_id}")
def get_results(analysis_id: str):
    """
    Get detailed analysis results
    
    Args:
        analysis_id: ID of the analysis
        
    Returns:
        Complete analysis results
    """
    if analysis_id not in analysis_results:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    
    return analysis_results[analysis_id]

@app.get("/api/analyses")
def list_analyses():
    """List all completed analyses"""
    analyses = []
    for analysis_id, results in analysis_results.items():
        analyses.append({
            "analysis_id": analysis_id,
            "timestamp": results.get("analysis_metadata", {}).get("timestamp"),
            "files_processed": results.get("file_statistics", {}).get("total_files", 0),
            "duplicates_found": len(results.get("similarity_analysis", {}).get("similar_pairs", [])),
            "status": results.get("analysis_metadata", {}).get("status", "unknown")
        })
    
    return {
        "total_analyses": len(analyses),
        "analyses": sorted(analyses, key=lambda x: x["timestamp"], reverse=True)
    }

@app.delete("/api/results/{analysis_id}")
def delete_analysis(analysis_id: str):
    """Delete an analysis result"""
    if analysis_id not in analysis_results:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {analysis_id} not found"
        )
    
    del analysis_results[analysis_id]
    return {
        "status": "✅ Analysis deleted",
        "analysis_id": analysis_id
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    print(f"❌ Unhandled exception: {exc}")
    print(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    print("\n" + "="*70)
    print("🟢 GreenBit Backend Started")
    print("="*70)
    print(f"✅ ML Pipeline: {'Available' if ML_PIPELINE_AVAILABLE and pipeline else 'Not Available'}")
    print(f"✅ CORS Enabled: {len(origins)} origins")
    print(f"✅ Documentation: /docs (Swagger) and /redoc (ReDoc)")
    print("="*70 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    print("\n" + "="*70)
    print("🔴 GreenBit Backend Shutdown")
    print("="*70 + "\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

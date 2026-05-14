"""
Test script for GreenBit API
Tests all endpoints with real file uploads
"""

import requests
import json
from pathlib import Path
import time

BASE_URL = "http://localhost:8000"

class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_section(title):
    """Print a formatted section"""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"{title}")
    print(f"{'='*70}{Colors.END}\n")

def print_success(msg):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg):
    """Print error message"""
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ️ {msg}{Colors.END}")

def test_health():
    """Test health endpoint"""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(f"  Status: {data['status']}")
            print(f"  ML Pipeline: {data['ml_pipeline_available']}")
            print(f"  Environment: {data['environment']}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_info():
    """Test info endpoint"""
    print_section("TEST 2: API Info")
    
    try:
        response = requests.get(f"{BASE_URL}/api/info")
        
        if response.status_code == 200:
            data = response.json()
            print_success("API info retrieved")
            print(f"  Name: {data['name']}")
            print(f"  Version: {data['version']}")
            print(f"  ML Pipeline: {data['ml_pipeline_status']}")
            return True
        else:
            print_error(f"Info request failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Info request error: {e}")
        return False

def test_analyze():
    """Test file analysis endpoint"""
    print_section("TEST 3: File Analysis")
    
    # Find sample files
    sample_files_dir = Path("../data/sample_files")
    
    if not sample_files_dir.exists():
        print_error(f"Sample files directory not found: {sample_files_dir}")
        return False, None
    
    files = list(sample_files_dir.glob("*.txt"))[:5]  # First 5 files
    
    if not files:
        print_error("No sample files found")
        return False, None
    
    print_info(f"Found {len(files)} sample files")
    
    try:
        # Prepare files for upload
        upload_files = [
            ("files", (f.name, open(f, "rb")))
            for f in files
        ]
        
        print_info("Uploading files for analysis...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/analyze",
            files=upload_files
        )
        
        elapsed_time = time.time() - start_time
        
        # Close file handles
        for _, (_, file_obj) in upload_files:
            file_obj.close()
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Analysis completed in {elapsed_time:.2f}s")
            print(f"  Analysis ID: {data['analysis_id']}")
            print(f"  Files processed: {data['files_processed']}")
            print(f"  Total size: {data['total_size_mb']:.2f} MB")
            print(f"  Duplicates found: {data['duplicates_found']}")
            print(f"  Clusters found: {data['clusters_found']}")
            print(f"  CO2 emissions: {data['co2_emissions_kg']:.6f} kg")
            
            return True, data['analysis_id']
        else:
            print_error(f"Analysis failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False, None
    except Exception as e:
        print_error(f"Analysis error: {e}")
        return False, None

def test_get_results(analysis_id):
    """Test get results endpoint"""
    print_section("TEST 4: Get Analysis Results")
    
    if not analysis_id:
        print_error("No analysis ID provided")
        return False
    
    try:
        response = requests.get(f"{BASE_URL}/api/results/{analysis_id}")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Results retrieved successfully")
            
            # Print summary
            print(f"\n📊 Analysis Summary:")
            
            if "file_statistics" in data:
                fs = data["file_statistics"]
                print(f"  Files analyzed: {fs.get('total_files', 0)}")
                print(f"  Unique files: {fs.get('unique_files', 0)}")
            
            if "clustering_results" in data:
                cr = data["clustering_results"]
                print(f"  Clusters found: {cr.get('total_clusters', 0)}")
                print(f"  Duplicate groups: {cr.get('duplicate_groups', 0)}")
            
            if "similarity_analysis" in data:
                sa = data["similarity_analysis"]
                print(f"  Similar pairs: {sa.get('similar_pairs_found', 0)}")
                
                # Print top similar pairs
                if sa.get("top_similar"):
                    print(f"\n🔗 Top Similar Pairs:")
                    for i, pair in enumerate(sa["top_similar"][:3], 1):
                        f1 = Path(pair["file1"]).name
                        f2 = Path(pair["file2"]).name
                        sim = pair["percentage"]
                        print(f"  {i}. {f1} <-> {f2}: {sim:.1f}%")
            
            if "carbon_emissions" in data:
                ce = data["carbon_emissions"]
                print(f"\n💨 Carbon Emissions:")
                print(f"  Inference: {ce.get('inference_emissions_formatted', 'N/A')}")
                print(f"  Status: {ce.get('tracking_status', 'N/A')}")
            
            if "environmental_impact" in data:
                ei = data["environmental_impact"]
                print(f"\n🌱 Environmental Impact:")
                print(f"  Storage freed: {ei.get('storage_freed_gb', 0):.2f} GB")
                print(f"  CO2 saved: {ei.get('co2_saved_annually_kg', 0):.6f} kg")
                print(f"  ROI: {ei.get('roi_percentage', 0):.0f}x")
            
            return True
        else:
            print_error(f"Get results failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Get results error: {e}")
        return False

def test_list_analyses():
    """Test list analyses endpoint"""
    print_section("TEST 5: List Analyses")
    
    try:
        response = requests.get(f"{BASE_URL}/api/analyses")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Retrieved {data['total_analyses']} analyses")
            
            if data['analyses']:
                print(f"\n📋 Recent Analyses:")
                for analysis in data['analyses'][:3]:
                    print(f"  • {analysis['analysis_id']}")
                    print(f"    Files: {analysis['files_processed']}, Duplicates: {analysis['duplicates_found']}")
            
            return True
        else:
            print_error(f"List analyses failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"List analyses error: {e}")
        return False

def test_delete_analysis(analysis_id):
    """Test delete analysis endpoint"""
    print_section("TEST 6: Delete Analysis")
    
    if not analysis_id:
        print_error("No analysis ID provided")
        return False
    
    try:
        response = requests.delete(f"{BASE_URL}/api/results/{analysis_id}")
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Analysis deleted: {data['analysis_id']}")
            return True
        else:
            print_error(f"Delete failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Delete error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print(f"\n{Colors.BLUE}{'='*70}")
    print("GREENBIT API TEST SUITE")
    print(f"{'='*70}{Colors.END}")
    
    results = {
        "health": False,
        "info": False,
        "analyze": False,
        "results": False,
        "list": False,
        "delete": False
    }
    
    # Test health
    results["health"] = test_health()
    
    # Test info
    results["info"] = test_info()
    
    # Test analyze
    success, analysis_id = test_analyze()
    results["analyze"] = success
    
    if analysis_id:
        # Test get results
        results["results"] = test_get_results(analysis_id)
        
        # Test list analyses
        results["list"] = test_list_analyses()
        
        # Test delete (optional - commented out to keep data)
        # results["delete"] = test_delete_analysis(analysis_id)
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✅ PASSED{Colors.END}" if result else f"{Colors.RED}❌ FAILED{Colors.END}"
        print(f"  {test_name.upper()}: {status}")
    
    print(f"\n{Colors.GREEN}Total: {passed}/{total} tests passed{Colors.END}\n")
    
    if passed == total:
        print_success("All tests passed! API is working correctly!")
        return True
    else:
        print_error(f"{total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    run_all_tests()

"""
Main Pipeline for GreenBit
Orchestrates the complete workflow: ingestion, embedding, clustering, carbon tracking
"""

from embeddings import EmbeddingEngine
from clustering import ClusteringEngine
from carbon_audit import CarbonAudit
import json
from pathlib import Path
import time

class GreenBitPipeline:
    """Complete GreenBit analysis pipeline"""
    
    def __init__(self):
        """Initialize all components"""
        print("🚀 Initializing GreenBit Pipeline...")
        
        self.embedding_engine = EmbeddingEngine()
        self.clustering_engine = ClusteringEngine(eps=0.35, min_samples=1)
        self.carbon_audit = CarbonAudit()
        
        print("✅ Pipeline ready!")
    
    def analyze_directory(self, directory_path, max_files=None):
        """
        Full pipeline: analyze a directory for duplicates
        
        Args:
            directory_path: Path to directory to analyze
            max_files: Maximum number of files to process
            
        Returns:
            Dictionary with complete analysis results
        """
        print(f"\n{'='*70}")
        print(f"GREENBIT ANALYSIS: {directory_path}")
        print(f"{'='*70}")
        
        # Step 1: Get all files
        print(f"\n📁 Step 1: Scanning directory...")
        file_paths = list(Path(directory_path).rglob("*"))
        file_paths = [str(f) for f in file_paths if f.is_file()]
        
        if max_files:
            file_paths = file_paths[:max_files]
        
        print(f"✅ Found {len(file_paths)} files")
        
        if not file_paths:
            return {"error": "No files found in directory", "status": "failed"}
        
        # Step 2: Start carbon tracking
        print(f"\n🌍 Step 2: Starting carbon tracking...")
        self.carbon_audit.start_tracking()
        start_time = time.time()
        
        # Step 3: Generate embeddings
        print(f"\n📊 Step 3: Generating semantic embeddings...")
        embeddings, contents, metadata = self.embedding_engine.generate_embeddings(file_paths)
        embedding_stats = self.embedding_engine.get_embedding_stats(embeddings)
        
        # Step 4: Cluster similar files
        print(f"\n🎯 Step 4: Clustering similar files...")
        clusters, labels = self.clustering_engine.cluster_embeddings(embeddings)
        
        # Step 5: Find similar pairs
        print(f"\n🔗 Step 5: Analyzing similarity pairs...")
        similar_pairs, similarity_matrix = self.clustering_engine.calculate_similarity(
            embeddings, 
            threshold=0.75
        )
        
        # Step 6: Calculate carbon emissions
        print(f"\n💨 Step 6: Calculating carbon emissions...")
        emissions_data = self.carbon_audit.stop_tracking()
        elapsed_time = time.time() - start_time
        
        # Step 7: Calculate environmental impact
        print(f"\n🌱 Step 7: Computing environmental impact...")
        
        # Estimate storage saved (10% of duplicate files)
        duplicate_files = sum(1 for pair in similar_pairs)
        estimated_storage_saved = duplicate_files * 0.5  # Rough estimate: 0.5 MB per file
        
        environmental_gain = self.carbon_audit.calculate_net_environmental_gain(
            storage_saved_gb=estimated_storage_saved / 1024,  # Convert to GB
            inference_emissions_kg=emissions_data['emissions_kg_co2'],
            duration_years=1
        )
        
        # Step 8: Prepare results
        print(f"\n✅ Analysis complete!")
        
        cluster_summary = self.clustering_engine.get_cluster_summary(clusters)
        
        results = {
            "analysis_metadata": {
                "timestamp": emissions_data["timestamp"],
                "directory": directory_path,
                "duration_seconds": elapsed_time,
                "status": "completed"
            },
            "file_statistics": {
                "total_files": len(file_paths),
                "files_analyzed": len(embeddings),
                "unique_files": len(file_paths) - len(similar_pairs)
            },
            "clustering_results": {
                "total_clusters": cluster_summary['total_clusters'],
                "duplicate_groups": cluster_summary['duplicate_groups'],
                "files_in_duplicates": cluster_summary['files_in_duplicates'],
                "outlier_files": cluster_summary['outlier_files']
            },
            "similarity_analysis": {
                "similar_pairs_found": len(similar_pairs),
                "top_similar": similar_pairs[:10] if similar_pairs else []
            },
            "carbon_emissions": {
                "inference_emissions_kg": float(emissions_data['emissions_kg_co2']),
                "inference_emissions_formatted": self.carbon_audit.format_carbon_display(
                    emissions_data['emissions_kg_co2']
                ),
                "energy_consumed_kwh": float(emissions_data['energy_consumed_kwh']),
                "tracking_status": emissions_data['status']
            },
            "environmental_impact": {
                "storage_freed_gb": environmental_gain['storage_freed_gb'],
                "co2_saved_annually_kg": environmental_gain['storage_emissions_saved_kg'],
                "net_environmental_gain_kg": environmental_gain['net_environmental_gain_kg'],
                "roi_percentage": environmental_gain['roi_percentage'],
                "trees_equivalent": environmental_gain['co2_equivalent']['trees_needed'],
                "miles_driven_equivalent": environmental_gain['co2_equivalent']['miles_driven']
            },
            "embedding_statistics": embedding_stats,
            "clusters": clusters,
            "all_similar_pairs": similar_pairs
        }
        
        return results
    
    def print_report(self, results):
        """Print a formatted analysis report"""
        
        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        print(f"\n{'='*70}")
        print("GREENBIT ANALYSIS REPORT")
        print(f"{'='*70}")
        
        # File statistics
        fs = results['file_statistics']
        print(f"\n📊 FILE STATISTICS")
        print(f"  Total files: {fs['total_files']}")
        print(f"  Files analyzed: {fs['files_analyzed']}")
        print(f"  Unique files: {fs['unique_files']}")
        
        # Clustering
        cr = results['clustering_results']
        print(f"\n🎯 CLUSTERING RESULTS")
        print(f"  Total clusters: {cr['total_clusters']}")
        print(f"  Duplicate groups: {cr['duplicate_groups']}")
        print(f"  Files in duplicates: {cr['files_in_duplicates']}")
        print(f"  Outlier files: {cr['outlier_files']}")
        
        # Similarity
        sa = results['similarity_analysis']
        print(f"\n🔗 SIMILARITY ANALYSIS")
        print(f"  Similar pairs found: {sa['similar_pairs_found']}")
        if sa['top_similar']:
            print(f"  Top 3 similar pairs:")
            for i, pair in enumerate(sa['top_similar'][:3], 1):
                f1 = Path(pair['file1']).name
                f2 = Path(pair['file2']).name
                print(f"    {i}. {f1} <-> {f2}: {pair['percentage']:.1f}%")
        
        # Carbon
        ce = results['carbon_emissions']
        print(f"\n💨 CARBON EMISSIONS")
        print(f"  Inference emissions: {ce['inference_emissions_formatted']}")
        print(f"  Energy consumed: {ce['energy_consumed_kwh']:.6f} kWh")
        print(f"  Tracking status: {ce['tracking_status']}")
        
        # Environmental Impact
        ei = results['environmental_impact']
        print(f"\n🌱 ENVIRONMENTAL IMPACT")
        print(f"  Storage freed: {ei['storage_freed_gb']:.2f} GB")
        print(f"  CO2 saved annually: {self.carbon_audit.format_carbon_display(ei['co2_saved_annually_kg'])}")
        print(f"  Net environmental gain: {self.carbon_audit.format_carbon_display(ei['net_environmental_gain_kg'])}")
        print(f"  ROI: {ei['roi_percentage']:.0f}x")
        print(f"  Equivalent to {ei['trees_equivalent']:.1f} trees planted")
        
        print(f"\n{'='*70}")
        print("✅ Analysis complete!")
        print(f"{'='*70}\n")

# Test the pipeline
if __name__ == "__main__":
    pipeline = GreenBitPipeline()
    
    # Analyze sample files
    results = pipeline.analyze_directory("../data/sample_files", max_files=12)
    
    # Print report
    pipeline.print_report(results)
    
    # Save results to JSON
    if "error" not in results:
        with open("analysis_results.json", "w") as f:
            # Convert numpy arrays and other non-serializable objects
            json.dump(results, f, indent=2, default=str)
        print("\n💾 Results saved to analysis_results.json")

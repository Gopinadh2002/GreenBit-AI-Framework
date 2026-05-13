"""
Clustering Engine for GreenBit
Groups similar files using DBSCAN clustering on semantic embeddings
"""

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm

class ClusteringEngine:
    """Find and cluster similar files"""
    
    def __init__(self, eps=0.3, min_samples=2, metric='cosine'):
        """
        Initialize clustering with DBSCAN
        
        Args:
            eps: Maximum distance between points in same cluster
            min_samples: Minimum points to form a cluster
            metric: Distance metric ('cosine', 'euclidean', etc.)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        print(f"🔧 Clustering configured: eps={eps}, min_samples={min_samples}, metric={metric}")
    
    def cluster_embeddings(self, embeddings_dict):
        """
        Cluster similar files based on embeddings
        
        Args:
            embeddings_dict: Dictionary of file paths to embeddings
            
        Returns:
            Tuple of (clusters dict, labels array)
        """
        print(f"\n🎯 Clustering {len(embeddings_dict)} files...")
        
        file_paths = list(embeddings_dict.keys())
        embeddings_array = np.array([embeddings_dict[fp] for fp in file_paths])
        
        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_normalized = scaler.fit_transform(embeddings_array)
        
        # Perform clustering
        labels = self.dbscan.fit_predict(embeddings_normalized)
        
        # Group files by cluster
        clusters = {}
        for file_path, label in zip(file_paths, labels):
            cluster_id = f"outlier" if label == -1 else f"cluster_{label}"
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(file_path)
        
        # Count duplicates
        duplicate_count = sum(1 for cluster in clusters.values() if len(cluster) > 1)
        
        print(f"✅ Found {len(clusters)} clusters")
        print(f"   - Duplicate groups: {duplicate_count}")
        print(f"   - Outliers: {len(clusters.get('outlier', []))}")
        
        return clusters, labels
    
    def calculate_similarity(self, embeddings_dict, threshold=0.75):
        """
        Calculate similarity scores between all file pairs
        
        Args:
            embeddings_dict: Dictionary of embeddings
            threshold: Minimum similarity to report (0-1)
            
        Returns:
            Tuple of (similar_pairs list, similarity_matrix)
        """
        print(f"\n📊 Calculating pairwise similarities...")
        
        file_paths = list(embeddings_dict.keys())
        embeddings_array = np.array([embeddings_dict[fp] for fp in file_paths])
        
        # Calculate cosine similarity matrix
        similarity_matrix = 1 - cdist(embeddings_array, embeddings_array, metric='cosine')
        
        # Find similar pairs
        similar_pairs = []
        for i in range(len(file_paths)):
            for j in range(i+1, len(file_paths)):
                sim_score = similarity_matrix[i][j]
                if sim_score >= threshold:
                    similar_pairs.append({
                        'file1': file_paths[i],
                        'file2': file_paths[j],
                        'similarity': float(sim_score),
                        'percentage': float(sim_score * 100)
                    })
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"✅ Found {len(similar_pairs)} similar pairs (threshold: {threshold*100:.0f}%)")
        
        return similar_pairs, similarity_matrix
    
    def get_cluster_summary(self, clusters):
        """
        Get summary statistics about clusters
        
        Args:
            clusters: Dictionary of clusters
            
        Returns:
            Summary dictionary
        """
        return {
            'total_clusters': len(clusters),
            'total_files': sum(len(files) for files in clusters.values()),
            'files_in_duplicates': sum(len(files) for files in clusters.values() if len(files) > 1),
            'duplicate_groups': sum(1 for files in clusters.values() if len(files) > 1),
            'outlier_files': len(clusters.get('outlier', []))
        }

# Test the clustering engine
if __name__ == "__main__":
    from embeddings import EmbeddingEngine
    from pathlib import Path
    
    print("=" * 70)
    print("TESTING CLUSTERING ENGINE")
    print("=" * 70)
    
    engine = EmbeddingEngine()
    clustering = ClusteringEngine(eps=0.35, min_samples=1)
    
    # Test with sample files
    sample_files = [
        "../data/sample_files/document1_draft.txt",
        "../data/sample_files/document1_draft_v2.txt",
        "../data/sample_files/document1_final.txt",
        "../data/sample_files/machine_learning_basics.txt",
        "../data/sample_files/ml_advanced_topics.txt",
    ]
    
    available_files = [f for f in sample_files if Path(f).exists()]
    
    if available_files:
        print(f"\n📁 Found {len(available_files)} sample files")
        
        # Generate embeddings
        embeddings, _, _ = engine.generate_embeddings(available_files)
        
        # Cluster files
        clusters, labels = clustering.cluster_embeddings(embeddings)
        
        # Get similar pairs
        similar_pairs, sim_matrix = clustering.calculate_similarity(embeddings, threshold=0.80)
        
        # Print clusters
        print(f"\n📂 Cluster Details:")
        for cluster_id, files in clusters.items():
            print(f"  {cluster_id}: {len(files)} files")
            for f in files:
                filename = Path(f).name
                print(f"    - {filename}")
        
        # Print similar pairs
        if similar_pairs:
            print(f"\n🔗 Top Similar Pairs:")
            for i, pair in enumerate(similar_pairs[:5], 1):
                f1 = Path(pair['file1']).name
                f2 = Path(pair['file2']).name
                print(f"  {i}. {f1} <-> {f2}")
                print(f"     Similarity: {pair['percentage']:.2f}%")
        
        # Summary
        summary = clustering.get_cluster_summary(clusters)
        print(f"\n📈 Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Clustering Engine test complete!")

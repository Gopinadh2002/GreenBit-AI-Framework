"""
Embedding Engine for GreenBit
Converts file content into semantic vectors using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

class EmbeddingEngine:
    """Generate semantic embeddings for files"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize semantic embedding model
        
        Args:
            model_name: Pre-trained transformer model to use
        """
        print(f"🤖 Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # MiniLM produces 384-dimensional vectors
        self.model_name = model_name
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def get_file_content(self, file_path):
        """
        Read file content safely
        
        Args:
            file_path: Path to file
            
        Returns:
            File content (first 5000 chars) or error message
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()[:5000]  # Limit to 5000 chars for efficiency
                return content if content.strip() else "[Empty File]"
        except Exception as e:
            return f"[Error reading file: {str(e)}]"
    
    def get_file_metadata(self, file_path):
        """
        Extract file metadata
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with metadata
        """
        try:
            path_obj = Path(file_path)
            return {
                "filename": path_obj.name,
                "size_bytes": path_obj.stat().st_size,
                "extension": path_obj.suffix,
                "is_file": path_obj.is_file()
            }
        except:
            return {}
    
    def generate_embeddings(self, file_paths):
        """
        Generate semantic embeddings for multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Tuple of (embeddings dict, file_contents dict, metadata dict)
        """
        print(f"\n📊 Generating embeddings for {len(file_paths)} files...")
        
        embeddings = {}
        file_contents = {}
        metadata = {}
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            # Get file content
            content = self.get_file_content(file_path)
            file_contents[file_path] = content
            
            # Get metadata
            metadata[file_path] = self.get_file_metadata(file_path)
            
            # Generate embedding
            embedding = self.model.encode(content)
            embeddings[file_path] = embedding
        
        print(f"✅ Generated {len(embeddings)} embeddings!")
        return embeddings, file_contents, metadata
    
    def get_embedding_stats(self, embeddings_dict):
        """
        Get statistics about embeddings
        
        Args:
            embeddings_dict: Dictionary of embeddings
            
        Returns:
            Dictionary with statistics
        """
        if not embeddings_dict:
            return {}
        
        embeddings_array = np.array(list(embeddings_dict.values()))
        
        return {
            "total_embeddings": len(embeddings_dict),
            "embedding_dimension": embeddings_array.shape[1],
            "mean_magnitude": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            "max_magnitude": float(np.max(np.linalg.norm(embeddings_array, axis=1))),
            "min_magnitude": float(np.min(np.linalg.norm(embeddings_array, axis=1)))
        }

# Test the embedding engine
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING EMBEDDING ENGINE")
    print("=" * 70)
    
    engine = EmbeddingEngine()
    
    # Test with sample files
    sample_files = [
        "../data/sample_files/document1_draft.txt",
        "../data/sample_files/document1_draft_v2.txt",
        "../data/sample_files/document1_final.txt",
    ]
    
    # Check if files exist
    available_files = [f for f in sample_files if Path(f).exists()]
    
    if available_files:
        print(f"\n📁 Found {len(available_files)} sample files")
        embeddings, contents, meta = engine.generate_embeddings(available_files)
        
        # Print stats
        stats = engine.get_embedding_stats(embeddings)
        print(f"\n📈 Embedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Show similarity between first two
        if len(embeddings) >= 2:
            files = list(embeddings.keys())
            emb1 = embeddings[files[0]]
            emb2 = embeddings[files[1]]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"\n🔗 Similarity between first two files: {similarity:.2%}")
    else:
        print("⚠️ Sample files not found. Run from ml_pipeline folder with data folder nearby.")
    
    print("\n✅ Embedding Engine test complete!")

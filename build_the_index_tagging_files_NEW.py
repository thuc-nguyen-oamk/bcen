import os
import torch
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import chromadb
import logging
from typing import List, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
DB_PATH = "./my_media_vault_qwen"
SOURCE_DIR = "/content/GPT-Image-2"
BATCH_SIZE = 8  # Keep batch size manageable for VRAM

class MultimodalIndexer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "Qwen/Qwen3-VL-Embedding-2B"
        
        logger.info(f"Loading {self.model_name} to {self.device}...")
        
        # Import locally to avoid issues if not installed
        try:
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="media_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.db_path = DB_PATH

    def get_all_files(self, root_dir: str) -> List[str]:
        """Recursively get all media files."""
        files = []
        for r, _, filenames in os.walk(root_dir):
            for f in filenames:
                if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS):
                    files.append(os.path.join(r, f))
        return files

    def load_content(self, file_path: str) -> Optional[Image.Image]:
        """Load image or video frame."""
        try:
            if any(file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                return Image.open(file_path).convert('RGB')
            elif any(file_path.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                # For videos, just take the first frame as a representative
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return None
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}")
            return None

    def process_batch(self, file_batch: List[str]):
        """Process a batch of files and store embeddings."""
        valid_files = []
        valid_images = []

        # Load images
        for f in file_batch:
            img = self.load_content(f)
            if img is not None:
                valid_files.append(f)
                valid_images.append(img)

        if not valid_images:
            return

        try:
            # Prepare inputs
            # Qwen3-VL processor requires text argument even if empty
            text_prompts = [""] * len(valid_images)
            
            inputs = self.processor(
                text=text_prompts,
                images=valid_images,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device and cast to float16
            pixel_values = inputs['pixel_values'].to(self.device).to(torch.float16)
            image_grid_thw = inputs['image_grid_thw'].to(self.device)

            with torch.no_grad():
                # Get features
                outputs = self.model.get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw
                )
                
                # Handle output format
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings_tensor = outputs.last_hidden_state
                else:
                    embeddings_tensor = outputs

                # Pooling: Mean over sequence dimension (dim=1)
                # Check if tensor is 3D (batch, seq, dim) or 2D (batch, dim)
                if embeddings_tensor.dim() == 3:
                    pooled = embeddings_tensor.mean(dim=1)
                elif embeddings_tensor.dim() == 2:
                    # Already pooled or single item without seq dim
                    pooled = embeddings_tensor
                else:
                    # Unexpected shape, unsqueeze to make it 2D
                    pooled = embeddings_tensor.unsqueeze(0)

                # Normalize embeddings (L2 norm)
                # Ensure we normalize over the last dimension (feature dim)
                if pooled.dim() == 1:
                    pooled = pooled.unsqueeze(0)  # Make sure it's 2D (1, dim)
                
                # Normalize along the feature dimension (last dim)
                embeddings_normalized = torch.nn.functional.normalize(pooled, p=2, dim=-1)

            # Convert to list
            embeddings_list = embeddings_normalized.cpu().float().numpy().tolist()

            # Validate lengths match before adding to DB
            if len(valid_files) != len(embeddings_list):
                logger.error(f"Mismatch: {len(valid_files)} files vs {len(embeddings_list)} embeddings")
                # Truncate to match to prevent crash
                min_len = min(len(valid_files), len(embeddings_list))
                valid_files = valid_files[:min_len]
                embeddings_list = embeddings_list[:min_len]

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings_list,
                ids=[os.path.basename(f) + "_" + str(i) for i, f in enumerate(valid_files)], # Unique IDs
                metadatas=[{"file_path": f} for f in valid_files],
                documents=[f for f in valid_files] # Store path as document too
            )
            
            logger.info(f"Processed batch: {len(valid_files)} items")

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

    def index_directory(self, root_dir: str):
        """Main indexing loop."""
        all_files = self.get_all_files(root_dir)
        logger.info(f"Found {len(all_files)} files. Starting indexing...")

        for i in tqdm(range(0, len(all_files), BATCH_SIZE)):
            batch = all_files[i : i + BATCH_SIZE]
            self.process_batch(batch)

def main():
    indexer = MultimodalIndexer()
    indexer.index_directory(SOURCE_DIR)
    print(f"Indexing complete. Database saved at {DB_PATH}")

if __name__ == "__main__":
    main()

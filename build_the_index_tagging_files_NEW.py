import os
import torch
import chromadb
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# --- CONFIGURATION ---
DB_PATH = "./my_media_vault"
COLLECTION_NAME = "local_files"
MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B" 
SKIP_FPS = 1.0  # Process 1 frame per second of video
BATCH_SIZE = 8  # Lower this if you run out of VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalIndexer:
    def __init__(self):
        # 1. Initialize ChromaDB (Persistent)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Best for semantic search
        )
        
        # Store DB path for later use
        self.db_path = DB_PATH
        
        # 2. Load Model & Processor
        print(f"Loading {MODEL_ID} to {DEVICE}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16,  # Half-precision for speed/memory
            trust_remote_code=True,
            device_map="auto"
        ).eval()

    def get_video_frames(self, video_path):
        """Smart Skip: Extracts frames based on SKIP_FPS."""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            duration_frames = len(vr)
            native_fps = vr.get_avg_fps()
            
            # Select frame indices (e.g., if 30fps and SKIP_FPS=1, take every 30th frame)
            step = max(1, int(native_fps / SKIP_FPS))
            indices = list(range(0, duration_frames, step))
            
            # Cap at 32 frames to prevent memory overflow on long videos
            indices = indices[:32]
            return [Image.fromarray(vr[i].asnumpy()) for i in indices]
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None

    def process_batch(self, file_batch):
        """Processes a batch of images or videos into embeddings."""
        batch_content = []
        batch_ids = []
        batch_metadata = []

        for path in file_batch:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext in ['.mp4', '.mov', '.avi', '.mkv']:
                    frames = self.get_video_frames(path)
                    if frames:
                        # For videos, we average the frames into one 'content' vector
                        # Note: Qwen3-VL handles list of images natively
                        batch_content.append(frames) 
                        batch_ids.append(path)
                        batch_metadata.append({"type": "video", "path": path})
                
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                    img = Image.open(path).convert("RGB")
                    batch_content.append(img)
                    batch_ids.append(path)
                    batch_metadata.append({"type": "image", "path": path})
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

        if not batch_content:
            return

        # 3. Generate Embeddings
        with torch.no_grad():
            # The processor handles the heavy lifting of resizing/normalizing
            # Provide empty text prompts for each item in the batch
            text_prompts = [""] * len(batch_content)
            inputs = self.processor(images=batch_content, text=text_prompts, return_tensors="pt").to(DEVICE)
            
            # Cast pixel_values to float16 to match model dtype
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            
            # Extract only the necessary keys for get_image_features
            # to avoid passing conflicting kwargs like 'attention_mask'
            pixel_values = inputs["pixel_values"]
            image_grid_thw = inputs["image_grid_thw"]
            
            # Get features from the model
            outputs = self.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            
            # Extract embeddings from the output object
            # Qwen3-VL-Embedding returns BaseModelOutputWithDeepstackFeatures
            # The last_hidden_state contains the sequence embeddings
            # We need to pool over the sequence dimension to get a single vector per image
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence
            else:
                # Fallback: if outputs is already a tensor
                embeddings = outputs
            
            embeddings_list = embeddings.cpu().detach().numpy().tolist()

        # 4. Save to ChromaDB
        self.collection.add(
            embeddings=embeddings_list,
            ids=batch_ids,
            metadatas=batch_metadata
        )

    def index_directory(self, root_dir):
        """Walks through folder and processes files in batches."""
        all_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                all_files.append(os.path.join(root, f))

        print(f"Found {len(all_files)} files. Starting indexing...")
        
        for i in tqdm(range(0, len(all_files), BATCH_SIZE)):
            batch = all_files[i : i + BATCH_SIZE]
            self.process_batch(batch)


# --- EXECUTION ---
if __name__ == "__main__":
    indexer = MultimodalIndexer()
    indexer.index_directory("/content/GPT-Image-2")
    print(f"Indexing complete. Database saved at {DB_PATH}")

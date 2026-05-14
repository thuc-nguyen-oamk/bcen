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

            metadata={"hnsw:space": "cosine"} # Best for semantic search

        )

        

        # 2. Load Model & Processor

        print(f"Loading {MODEL_ID} to {DEVICE}...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(

            MODEL_ID, 

            torch_dtype=torch.float16, # Half-precision for speed/memory

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

                

                elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.pdf']:

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

            inputs = self.processor(images=batch_content, return_tensors="pt").to(DEVICE)

            # Use FP16 to match model weight types

            inputs = {k: v.to(torch.float16) if v.is_floating_point() else v for k, v in inputs.items()}

            

            embeddings = self.model.get_image_features(**inputs)

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

    indexer.index_directory("/content/test")

    print(f"Indexing complete. Database saved at {DB_PATH}")

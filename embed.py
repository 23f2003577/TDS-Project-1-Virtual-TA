# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "semantic-text-splitter",
#     "tqdm",
#     "google-genai",
#     "google-generativeai",
#     "python-dotenv",
#     "numpy",
#     "sentence-transformers"
# ]
# ///

import os
import numpy as np
from tqdm import tqdm
from semantic_text_splitter import MarkdownSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # Local model

def get_chunks(file_path, chunk_size=15000):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = MarkdownSplitter(chunk_size)
    return splitter.chunks(text)

def main(markdown_dir="markdowns", output_file="still_merged_embeddings.npz"):
    all_chunks = []
    all_embeddings = []

    files = [f for f in os.listdir(markdown_dir) if f.endswith(".md")]
    for fname in tqdm(files, desc="Generating embeddings"):
        path = os.path.join(markdown_dir, fname)
        chunks = get_chunks(path)
        for chunk in chunks:
            try:
                embedding = model.encode(chunk)
                all_chunks.append(chunk)
                all_embeddings.append(embedding)
            except Exception as e:
                print(f"❌ Error in {fname}: {e}")

    np.savez(output_file, chunks=all_chunks, embeddings=np.array(all_embeddings))
    print(f"✅ Saved {len(all_chunks)} chunks and embeddings to {output_file}")

if __name__ == "__main__":
    main()

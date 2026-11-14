import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer

ds = ir_datasets.load("beir/scifact/test")
doc_texts = [f"{d.title} {d.text}".strip() for d in ds.docs_iter()]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_emb = model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)
np.save("doc_emb.npy", doc_emb)
print("Saved doc_emb.npy successfully.")
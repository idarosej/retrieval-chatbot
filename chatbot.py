from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Knowledge base
documents = [
    "Python is a programming language.",
    "Machine learning enables computers to learn from data.",
    "GitHub is a platform for hosting code repositories."
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(documents)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=1)

    response = documents[indices[0][0]]
    print("Bot:", response)

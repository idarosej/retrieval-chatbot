from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load documents
with open("data.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

# Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = FAISS.from_texts(documents, embedding_model)

# Chat loop
while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=1)

    print("Bot:", docs[0].page_content)

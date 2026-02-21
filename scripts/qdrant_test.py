from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
print(client.get_collections())
info = client.get_collection("rag_docs")
print(f"Points: {info.points_count}")
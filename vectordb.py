import chromadb
import numpy as np   

# client = chromadb.PersistentClient(path="vectordb") 

def store_in_vectordb(chunks, vectors, path="vectordb", name="document_chunks"):
    client = chromadb.PersistentClient(path=path)

    # Check if the collection already exists
    try:
        collection = client.get_collection(name=name)
    except ValueError:
        # If the collection doesn't exist, create it
        collection = client.create_collection(name=name)

    # Prepare data for batch insertion
    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"chunk": chunk} for chunk in chunks]
    embeddings = [vector.tolist() for vector in vectors]
    
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print("Chunks and vectors stored in ChromaDB")

    
def retrieve_top_similar_chunks_from_vectordb(query_embed, path, name):
    client = chromadb.PersistentClient(path=path)
    
    try:
        collection = client.get_collection(name=name)
    except ValueError:
        raise ValueError(f"Collection '{name}' does not exist.")

    try:
        items = collection.peek()
        all_vectors = items['embeddings']  # Extract embeddings (vectors)
        all_metadatas = items['metadatas']  # Extract metadata containing chunks
        all_chunks = [metadata['chunk'] for metadata in all_metadatas]
    except AttributeError as e:
        raise AttributeError("'Collection' object does not support 'peek'. Check the ChromaDB documentation for the correct method.") from e
    except KeyError as e:
        raise KeyError(f"Missing expected key in the items: {e}. Check the structure of the returned data.") from e

    all_vectors = np.array(all_vectors)

    # Compute similarities between the query and all vectors
    similarities = np.dot(all_vectors, query_embed.T)
    top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()
    most_similar_chunks = [all_chunks[idx] for idx in top_3_idx]

    return most_similar_chunks




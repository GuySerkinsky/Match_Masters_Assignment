import shutil  
import textwrap
import numpy as np
from utils import fetch_pdf_chunks
from sentence_transformers import SentenceTransformer
from vectordb import store_in_vectordb, retrieve_top_similar_chunks_from_vectordb

def main():
    # Fetch chunks from PDFs file
    chunks = fetch_pdf_chunks()
    print(f"Total chunks fetched: {len(chunks)}")
    
    model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
        

    if chunks:
     # Embed the Chunks
     docs_vectors = model.encode(chunks, normalize_embeddings=True)
     print(f"First embedded chunk: {docs_vectors[0]}")
     # New vector DB can be created by modifying the "path" value, and new collections can be added to an existing database by changing the "name" value.
     store_in_vectordb(chunks, docs_vectors, "vectordb", "document_chunks")

    while True:
        # Embed the Query
       query = input("Enter your query >>> ")
       if query.lower() == 'exit':
            break
        
       query_embed = model.encode(query, normalize_embeddings=True)
       print(f"Embedded query: {query_embed}")

       # Retrieve most similar chunks from vectordb path="name of vectordb" name="collection name"
       similarities = retrieve_top_similar_chunks_from_vectordb(query_embed, "vectordb", "document_chunks")

       top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()
       print(f"\nTop 3 similarity chunks (indexes): {top_3_idx}")


       RELEVANT_CHUNKS = ""
       for i, p in enumerate(similarities):
           wrapped_text = textwrap.fill(p, width=100)
           RELEVANT_CHUNKS += wrapped_text + "\n\n"
    
       print(f"Most similar chunks:\n\n {RELEVANT_CHUNKS}")

if __name__ == "__main__":
    main()

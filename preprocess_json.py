import requests
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(texts):
    import requests

    url = "http://localhost:11434/api/embed"
    embeddings = []

    for i, text in enumerate(texts, start=1):
        response = requests.post(
            url,
            json={
                "model": "bge-m3:latest",
                "input": text
            },
            timeout=300
        )

        if not response.ok:
            print(f"Failed at chunk {i}")
            print(f"Text length: {len(text)} characters")
            print("Ollama error:", response.text)
            response.raise_for_status()

        result = response.json()
        embeddings.append(result["embeddings"][0])

        if i % 10 == 0:
            print(f"Embedded {i}/{len(texts)} chunks")

    return embeddings


new_jsons = os.listdir("new_jsons") #List all the new_jsons
my_dicts = []
chunk_id = 0

for json_file in new_jsons:
    with open(f"new_jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# save this dataframe
joblib.dump(df, 'embedding.joblib')


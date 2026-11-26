import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib
import numpy as np


# Generate embeddings using the local BGE-M3 model
def create_embedding(texts):
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


# Generate an answer using the local Llama 3.2 model
def inference(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            # "model": "deepseek-r1",
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()
    return response.json()["response"]


# Load the saved DataFrame containing lecture chunks and embeddings
df = joblib.load("embedding.joblib")


# Get the user's question and generate its embedding
incoming_query = input("Ask a Question:")
question_embedding = create_embedding([incoming_query])[0]


# Retrieve the most similar lecture chunks using cosine similarity
similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()

top_results = 3
max_index = similarities.argsort()[::-1][0:top_results]

# Select the matching rows by their positional indices
new_df = df.iloc[max_index].reset_index(drop=True)

# Convert seconds into minutes:seconds format
def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Convert retrieved chunk timestamps to MM:SS
new_df = new_df.copy()
new_df["start"] = new_df["start"].apply(format_timestamp)
new_df["end"] = new_df["end"].apply(format_timestamp)


# Build the prompt using the retrieved lecture chunks
prompt = f"""
You are a Statistics Teaching Assistant.
Answer the user's question using the retrieved lecture context provided below.

Follow these rules carefully:

1. Use the retrieved lecture excerpts as the primary and only source of lecture-specific information.
2. Explain concepts clearly and in simple language.
3. Do not invent definitions, formulas, examples, or explanations.
4. If the retrieved excerpts provide only partial information, explain only what is supported and mention what is missing.
5. Do not repeat the same explanation in different words.
6. Do not interpret mathematical notation beyond what the lecture supports.
7. Preserve mathematical symbols and statistical notation accurately.
8. Do not include greetings or unnecessary introductions.
9. Do not invent lecture titles, numbers, or timestamps.
10. Do not present general statistical knowledge as if it were explicitly stated in the lecture.

Use this response format:

**Answer:**
Provide a concise explanation based on the retrieved excerpts.

**Relevant Lecture Content:**
Include only information supported by the retrieved lecture excerpts.
 Do not introduce unrelated book references or external information as lecture content.


**Source:**
   - Provide the lecture title, lecture number, and relevant timestamp.
   - Always provide the lecture title, lecture number, and relevant timestamp(s).
   - Use only the lecture metadata and timestamps available in the retrieved context.
   - Include all relevant timestamps that support the answer.
   - Do not invent or estimate missing source details.
   - Do not write "timestamps not specified" when timestamps are available.

Insufficient Information:
   - If the retrieved excerpts are insufficient to answer the question fully, clearly state:
     "The available lecture excerpts do not contain enough information to fully answer this question."

If the retrieved excerpts are insufficient, clearly state:
"The available lecture excerpts do not contain enough information to fully answer this question."

 Output Formatting:
   - Never copy the raw retrieved context or JSON directly into the response.
   - Convert retrieved information into clear, readable bullet points or sentences.
   - In the Relevant Lecture Content section, summarize the relevant lecture excerpts without changing their meaning.
   - In the Source section, list only the lecture title, lecture number, and timestamp.
   - Do not repeat the full transcript text in the Source section.
   - Follow the same three-section format for every question.

Retrieved Lecture Context:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

User Question:
{incoming_query}

Answer:
"""


# Save the prompt for inspection
with open("prompt.txt", "w") as f:
    f.write(prompt)


# Generate and display the answer
response = inference(prompt)
print(response)


# Save the answer as UTF-8 text
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
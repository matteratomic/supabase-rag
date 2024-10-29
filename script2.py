import os
import re
import json
from transformers import AutoTokenizer
from docx import Document
from openai import OpenAI
import psycopg2
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Load the tokenizer once to avoid reloading it multiple times
tokenizer = AutoTokenizer.from_pretrained("baai/bge-base-en-v1.5")

# Connect to PostgreSQL
def connect_db():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )

def extract_year(text):
    match = re.search(r'\b\d{4}\b', text)
    if match:
        return str(match.group())

    return None

def process_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunk_size = 768
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    text_chunks = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
    return text_chunks

def get_embedding(text, model="compendiumlabs/bge-base-en-v1.5-gguf"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def save_chunks_to_db(cursor, chunks):
    chunk_values = [(chunk,) for chunk in chunks]
    cursor.executemany('INSERT INTO text_chunks (text_chunk) VALUES (%s) RETURNING id', chunk_values)
    return [row[0] for row in cursor.fetchall()]

def save_chunks_and_link_embedding(cursor, text):
    text_chunks = process_text(text)
    
    # Parallelize embedding process
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(get_embedding, text_chunks))
    
    ids = save_chunks_to_db(cursor, text_chunks)
    return list(zip(ids, embeddings))

def process_directory(directory):
    embeddings = []
    
    with connect_db() as conn:
        with conn.cursor() as cursor:
            for dirpath, dirnames, files in os.walk(directory):
                for file in files:
                    if file.endswith(".docx"):
                        file_path = os.path.join(dirpath, file)
                        print(f'Processing {file_path}')
                        file_year = extract_year(file_path)
                        text = extract_text_from_docx(file_path)
                        linked_data = save_chunks_and_link_embedding(cursor, text)

                        relative_path = os.path.relpath(dirpath, directory)
                        path_parts = relative_path.split(os.sep)

                        state = path_parts[0] if len(path_parts) >= 1 else "Unknown"
                        city = path_parts[1] if len(path_parts) > 1 else "N/A"

                        for id, embedding in linked_data:
                            embeddings.append({
                                "metadata": {
                                    "id": id,
                                    "file_path": file_path,
                                    "year": file_year,
                                    "state": state,
                                    "city": city
                                }
                            })
            conn.commit()
    
    return embeddings

root_directory = r".\codes"
all_embeddings = process_directory(root_directory)

# Save embeddings to a JSON file (in chunks to reduce memory usage)
with open('embeddings.json', 'w') as f:
    json.dump(all_embeddings, f)

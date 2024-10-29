import os
import re
from transformers import AutoTokenizer
from docx import Document
from openai import OpenAI
import psycopg2
from supabase import create_client

# url = os.environ.get("http://localhost:8000")
# key = os.environ.get("SUPABASE_CLIENT_API_KEY")
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE"
supabase = create_client("http://localhost:8000", key)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def connect_db():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )

tokenizer = AutoTokenizer.from_pretrained("baai/bge-base-en-v1.5")


def save_to_db(embedding, content, state, city, year, path):
    response = supabase.table('documents').insert({
        "embedding": embedding,
        "content": content,
        "state": state,
        "city": city,
        "year": year,
        "path":path
    }).execute()
    return response


def process_text(text):
    # Here you can add any text cleaning or chunking logic
    # For simplicity, we're just returning the text as is

    # Tokenize the document into tokens
    tokens = tokenizer(text, return_tensors="pt",
                       truncation=False)["input_ids"][0]

    # Split into chunks of 768 tokens
    chunk_size = 768
    chunks = [tokens[i:i + chunk_size]
              for i in range(0, len(tokens), chunk_size)]

    # Decode each chunk back to text (optional, for human readability)
    text_chunks = [tokenizer.decode(
        chunk, skip_special_tokens=True) for chunk in chunks]
    embeddings = [get_embedding(chunk) for chunk in text_chunks]
    return zip(text_chunks, embeddings)


# def get_embedding(text, model="compendiumlabs/bge-base-en-v1.5-gguf"):
def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])


def save_chunk_to_db(chunk, cursor):
    cursor.execute(
        'INSERT INTO text_chunks (text_chunk) VALUES (%s) RETURNING id', (chunk,))
    return cursor.fetchone()[0]


def save_chunks_and_link_embedding(text, cursor):
    processed_data = process_text(text)
    linked_data = []
    for chunk, embedding in processed_data:
        id = save_chunk_to_db(chunk, cursor)
        linked_data.append((id, embedding))
    return linked_data


def extract_year(text):
    match = re.search(r'\b\d{4}\b', text)
    if match:
        return str(match.group())

    return None


def process_directory(directory):
    for dirpath, dirnames, files in os.walk(directory):
        for file in files:
            if file.endswith(".docx"):
                file_path = os.path.join(dirpath, file)
                print(f'Processing {file_path}')
                file_year = extract_year(file_path)
                text = extract_text_from_docx(file_path)
                relative_path = os.path.relpath(dirpath, directory)
                path_parts = relative_path.split(os.sep)

                if len(path_parts) == 1:  # Only the state folder
                    state = path_parts[0]
                    city = "N/A"
                elif len(path_parts) > 1:  # There is a city folder inside the state folder
                    state = path_parts[0]
                    city = path_parts[1]
                else:
                    state = "Unknown"
                    city = "Unknown"

                processed_data = process_text(text)
                for chunk, embedding in processed_data:
                    save_to_db(
                        path=file_path,
                        embedding=embedding,
                        content=chunk,
                        state=state,
                        city=city,
                        year=file_year
                    )

# Specify the root directory containing your state folders
root_directory = r".\states"

# Process all documents and generate embeddings
process_directory(root_directory)
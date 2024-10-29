import os
import re
import json
from transformers import AutoTokenizer
from docx import Document
from openai import OpenAI
import psycopg2
from supabase import create_client


# from cloudflare_vectorize import CloudflareVectorize  # You'll need to implement this
# url = os.environ.get("http://localhost:8000")
# key = os.environ.get("SUPABASE_CLIENT_API_KEY")
key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE"
supabase = create_client("http://localhost:8000", key)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# conn = psycopg2.connect(
#     dbname="postgres",
#     user="postgres",
#     password="root",
#     host="localhost",
#     port="5432"
# )

def connect_db():
    return psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )


# cursor = conn.cursor()
tokenizer = AutoTokenizer.from_pretrained("baai/bge-base-en-v1.5")


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


def get_embedding(text, model="compendiumlabs/bge-base-en-v1.5-gguf"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])


def save_chunk_to_db(chunk,cursor):
    cursor.execute(
        'INSERT INTO text_chunks (text_chunk) VALUES (%s) RETURNING id', (chunk,))
    return cursor.fetchone()[0]

def save_chunks_and_link_embedding(text,cursor):
    processed_data = process_text(text)
    linked_data = []
    for chunk, embedding in processed_data:
        id = save_chunk_to_db(chunk,cursor)
        linked_data.append((id, embedding))
    return linked_data


def extract_year(text):
    match = re.search(r'\b\d{4}\b', text)
    if match:
        return str(match.group())

    return None


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
                        linked_data = save_chunks_and_link_embedding(text,cursor)
                        # print(f'This is the linked data {len(linked_data)}')
                        # print(f'File path is {file_path}. Dirpath:{dirpath} The relative path is { relative_path} and directory is {directory}')
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

                        for id, embedding in linked_data:
                            # print(f'This is the id {id}')
                            embeddings.append({
                                "id": id,
                                "values": embedding,
                                "namespace":state,
                                "metadata": {
                                    "file_path": file_path,
                                    "year": file_year,
                                    "state": state,
                                    "city": city
                                }
                            })
    conn.commit()
    # cursor.close()
    # conn.close()
    return embeddings


# Specify the root directory containing your state folders
root_directory = r".\codes"

# Process all documents and generate embeddings
# all_embeddings = process_directory(root_directory)

# Save embeddings to a JSON file (optional, for backup)
# with open('embeddings.json', 'w') as f:
#     json.dump(all_embeddings, f)
    

def add_to_supabase(embedding,content):
    response = supabase.table('documents').insert({
        "embedding":embedding,
        "content":content
    }).execute()

    print(str(response))
    # if response.status_code == 201:
    #     print("INSERT SUCCESSFULLY")
    #     print(response.data)
    # else:
    #     print("INSERT FAILED")
    #     print(response.error)


add_to_supabase()


# interface VectorizeVector {
#   /** The ID for the vector. This can be user-defined, and must be unique. It should uniquely identify the object, and is best set based on the ID of what the vector represents. */
#   id: string;
#   /** The vector values */
#   values: VectorFloatArray | number[];
#   /** The namespace this vector belongs to. */
#   namespace?: string;
#   /** Metadata associated with the vector. Includes the values of other fields and potentially additional details. */
#   metadata?: Record<string, VectorizeVectorMetadata>;
# }

# # Initialize Cloudflare Vectorize client
# vectorize = CloudflareVectorize(api_key="your_api_key", account_id="your_account_id", index_name="your_index_name")

# # Store embeddings in Cloudflare Vectorize
# for item in all_embeddings:
#   vectorize.upsert(
#       id=item["metadata"]["file_path"],
#       vector=item["embedding"],
#       metadata=item["metadata"]
#   )

# print("Embeddings generated and stored in Cloudflare Vectorize")
# Created/Modified files during execution:
# print(str(all_embeddings))

# create or replace function match_documents (
#   query_embedding vector(768),
#   match_threshold float,
#   match_count int
# )  
# returns table (
#     id bigint,
#     content text,
#     similarity float,
#   )
# language sql stable
#   as $$
#     SELECT 
#     documents.id,
#     documents.content,
#     1 - ( documents.embedding <=> query_embedding) AS similarity 
#     FROM documents 
#     WHERE (documents.embedding <=> query_embedding) < 1 - match_threshold
#     ORDER BY documents.embedding <=> query_embedding
#     LIMIT match_count;
#   $$;
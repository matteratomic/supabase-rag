from openai import OpenAI
from supabase import create_client

key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE"

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
supabase = create_client("http://localhost:8000", key)


def get_completion(query, context):
    completion = client.chat.completions.create(
        model="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who helps users with information they request."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": query}
        ],
        temperature=0.7,
    )
    print(completion.choices[0].message.content)


def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def search_db(query):
    query_embedding = get_embedding(query)
    response = supabase.rpc('match_documents', {
        "city": None,
        "state": None,
        "year": None,
        "query_embedding": query_embedding,
        "match_threshold": 0.50,
        "match_count": 10,
    }).execute()
    # print(response)
    return response

# nomic_rules = """401.1 Minimum Floor Elevation of Storm Shelters"""

def RAG():
    query = "Complete the sentence:  where cable and pulleys are used to connect panels of multisection sliding doors, each pulley shall be equipped with a...."
    response = search_db(query)
    context = response.data[0]['content']
    print(context)
    print("*"*10)
    print("-"*10)
    get_completion(query, context)
RAG()
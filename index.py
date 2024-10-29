from openai import OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embedding(text, model="CompendiumLabs/bge-base-en-v1.5-gguf"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# print(get_embedding("What is my favorite food?"))
# file = open('array.txt','w+')
with open('array.txt','w+') as file:
    file.write(str(get_embedding("What is the secret code?")))

input("Press any key to exit")
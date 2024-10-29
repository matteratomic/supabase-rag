import os
def process_directory(directory):
  embeddings = []
  for dirpath, dirnames, filenames in os.walk(directory):
      for file in filenames:
          if file.endswith(".docx"):
              file_path = os.path.join(dirpath, file)
            #   text = extract_text_from_docx(file_path)
            #   processed_text = process_text(text)
            #   embedding = get_embedding(processed_text)

              embeddings.append({
                  "metadata": {
                    #   "file_path": file_path,
                    # "embedding": embedding,
                      "state": os.path.basename(os.path.dirname(dirpath)),
                      "city": os.path.basename(dirpath) if dirpath != directory else "N/A"
                  }
              })
  return embeddings

result = process_directory('./codes')
print(str(result))
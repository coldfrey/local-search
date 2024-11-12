from sentence_transformers import SentenceTransformer
import os
import numpy as np

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Directory containing note files
notes_directory = 'notes/'

# Iterate over each file in the directory
for filename in os.listdir(notes_directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(notes_directory, filename)
        
        # Read the contents of the file
        with open(file_path, 'r') as file:
            content = file.read().strip()
        
        # Encode the content of the file
        embedding = model.encode(content)
        # save the embedding to a file in the embeddings directory
        embedding_path = os.path.join('embeddings', filename.replace('.txt', '.npy'))
        with open(embedding_path, 'wb') as file:
            np.save(file, embedding)
        print(f'Embedding saved to {embedding_path}')

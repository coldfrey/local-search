from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from scipy.spatial.distance import cosine

app = Flask(__name__)
CORS(app)

# Load the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Directory paths
notes_directory = 'notes/'
embeddings_directory = 'embeddings/'

# Load all precomputed embeddings and notes
notes = []
embeddings = []

for filename in os.listdir(notes_directory):
    if filename.endswith('.txt'):
        # Load the content of the note
        file_path = os.path.join(notes_directory, filename)
        with open(file_path, 'r') as file:
            content = file.read().strip()
        notes.append({'filename': filename, 'content': content, 'title': filename.replace('.txt', '')})
        
        # Load the corresponding embedding
        embedding_path = os.path.join(embeddings_directory, filename.replace('.txt', '.npy'))
        embedding = np.load(embedding_path)
        embeddings.append(embedding)

# Extract note contents for similarity computation
note_contents = [note['content'] for note in notes]

@app.route('/search', methods=['POST'])
def search_similar_notes():
    data = request.json
    user_message = data.get('message', '')
    user_embedding = model.encode(user_message)
    similarities = [1 - cosine(user_embedding, emb) for emb in embeddings]
    most_similar_index = np.argmax(similarities)
    most_similar_note = note_contents[most_similar_index]
    similarity_score = similarities[most_similar_index]
    # filename without the .txt extension
    most_similar_filename = notes[most_similar_index]['filename']

    return jsonify({
        'filename': most_similar_filename,
        'most_similar_note': most_similar_note,
        'similarity_score': similarity_score
    }), 200

@app.route('/notes', methods=['GET'])
def get_notes():
    note_list = [{'filename': note['filename'], 'title': note['title']} for note in notes]
    return jsonify(note_list), 200

@app.route('/note/<filename>', methods=['GET'])
def get_note_content(filename):
    if not filename.endswith('.txt'):
        filename += '.txt'
    file_path = os.path.join(notes_directory, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            content = file.read().strip()
        return jsonify({'content': content}), 200
    else:
        return jsonify({'error': 'Note not found'}), 404

@app.route('/add_note', methods=['POST'])
def add_note():
    data = request.json
    content = data.get('content', '')
    if not content:
        return jsonify({'error': 'Content is required'}), 400

    # Generate a filename for the new note
    filename = f'note_{len(notes) + 1}.txt'
    file_path = os.path.join(notes_directory, filename)

    # Save the content to a new file
    with open(file_path, 'w') as file:
        file.write(content)

    # Compute the embedding for the new note
    embedding = model.encode(content)
    embedding_path = os.path.join(embeddings_directory, filename.replace('.txt', '.npy'))
    with open(embedding_path, 'wb') as file:
        np.save(file, embedding)

    # Add the new note to the notes list
    notes.append({'filename': filename, 'content': content, 'title': filename.replace('.txt', '')})
    embeddings.append(embedding)

    return jsonify({'message': 'Note added successfully'}), 201


if __name__ == '__main__':
    app.run(debug=True)

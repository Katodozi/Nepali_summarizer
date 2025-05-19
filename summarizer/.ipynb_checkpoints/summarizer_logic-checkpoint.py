import numpy as np

# Define the path to your embeddings file
embeddings_file_path = r"C:\Users\Anuz\OneDrive\Desktop\excel work\Embeddings.txt"

# Initialize an empty dictionary to store embeddings
embeddings = {}

# Read the embeddings from the file
with open(embeddings_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()  # Split line into word and vector components
        word = parts[0]  # The first part is the word
        vector = np.array(parts[1:], dtype=float)  # Convert the rest to a NumPy array of floats
        embeddings[word] = vector  # Store in the dictionary

print(f"Loaded {len(embeddings)} word embeddings.")

import pandas as pd

# Define the path to your tokenized data file
tokenized_data_file_path = r"C:\Users\Anuz\OneDrive\Desktop\excel work\stemmed_tokenized_cleaned_dataset.csv"

# Load the tokenized data from CSV file
df = pd.read_csv(tokenized_data_file_path)

# Convert string representation of lists back to actual lists
tokenized_sentences = df['stemmed_text'].apply(eval).tolist()

print(f"Loaded {len(tokenized_sentences)} tokenized sentences.")

# Create a vocabulary mapping from words to indices
word_to_index = {word: idx for idx, word in enumerate(embeddings.keys())}
index_to_word = {idx: word for word, idx in word_to_index.items()}

print(f"Vocabulary size: {len(word_to_index)}")
vocab_dict = word_to_index

#implementing the textrank algorithm
from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_vector(sentence, embeddings):
    tokens = sentence.split()  # Tokenize the sentence into words
    vectors = [embeddings[word] for word in tokens if word in embeddings]
    
    # Return a zero vector if no valid tokens are found
    if not vectors:  
        return np.zeros(100)  # Return a zero vector of embedding size (100)
    
    return np.mean(vectors, axis=0)  # Average out the vectors
def textrank(sentences, embeddings):
    sentence_vectors = []
    
    for sentence in sentences:
        vector = get_sentence_vector(sentence, embeddings)
        # print(f"Sentence: {sentence}")  # Print the sentence
        # print(f"Vector shape: {vector.shape}")  # Print the shape of the vector
        sentence_vectors.append(vector)

    #Convert list of vectors to a NumPy array
    try:
        sentence_vectors = np.array(sentence_vectors)
    except ValueError as e:
        print("Error creating NumPy array:", e)
        print("Sentence Vectors:", sentence_vectors)  # Debugging output
        return []

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Rank sentences based on scores (sum of similarities)
    scores = np.sum(similarity_matrix, axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores)[::-1]]  # Sort sentences by score
    
    return ranked_sentences[:3]  # Return top 3 sentences as summary
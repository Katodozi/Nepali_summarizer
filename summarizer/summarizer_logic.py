import numpy as np
import os
from django.conf import settings  # Add this at the top

# Define the path relative to project root
embeddings_file_path = os.path.join(settings.BASE_DIR, 'data', 'Embeddings.txt')

#defining the path to our embeddings file
#embeddings_file_path = r"C:\Users\Anuz\OneDrive\Desktop\excel work\Embeddings.txt"

#initializing an empty dictionary to store embeddings
embeddings = {}

#reading the embeddings from the file
with open(embeddings_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()  #splitting the line into word and vector components
        word = parts[0]  #The first part is the word
        vector = np.array(parts[1:], dtype=float)  #Converting the rest to a NumPy array of floats
        embeddings[word] = vector  #storing in the dictionary

print(f"Loaded {len(embeddings)} word embeddings.")

#creating a vocabulary mapping from words to indices
word_to_index = {word: idx for idx, word in enumerate(embeddings.keys())}
index_to_word = {idx: word for word, idx in word_to_index.items()}

print(f"Vocabulary size: {len(word_to_index)}")
vocab_dict = word_to_index 

#implementing the textrank algorithm
from sklearn.metrics.pairwise import cosine_similarity

def nepali_stemmer(word):
    """Rule-based stemmer for Nepali words"""
    suffixes = [   
     'अर्को', 'बाट', 'बाहेक', 'बाहिर', 'बाहिरपट्टी',
    'भित्र', 'का', 'करिब', 'को', 'छ', 'छिन्',
    'जोड', 'ले', 'लागि',
    'लाई', 'माथि', 'मन्तिर', 'मा', 'नजिक',
    'पछाडि', 'पहिला', 'पारि', 'प्रति', 'र',
    'संग','सहित','तल','तर','तिर',
    'तर्फ','उपर','विपरित','वरिपरि','भित्र','बिचमा']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# Financial terms with boost weights (customize for Nepali finance)
FINANCIAL_BOOST = {
    # Terms        # Weight (1.5-3x)
    "खर्च": 2.3, # Expense
    "आम्दानी": 2.4, # Income
    "लगानी": 2.5, # Investment
    "बचत": 2.2, # Savings
    "मूल्य": 2.1, # Value/Price
    "विनिमय": 2.3, # Exchange
    "तलब": 2.0, # Salary
    "सम्पत्ति": 2.4, # Assets
    "जम्मा": 2.2, # Deposit
    "निकासी": 2.3, # Withdrawal
    "मुनाफा": 2.5, # Profit (alternative to "नाफा")
    "घाटा": 2.4, # Loss
}

# Regex for Nepali numbers/dates/currency
NUMERICAL_PATTERN = r'''
(रु\.\s?\d+[\s,]*\d*|       # रु. १०,०००
\d{4}\s[बैशाख|जेठ|असार|साउन|भदौ|असोज|कात्तिक|मंसिर|पुष|माघ|फाल्गुन|चैत]+\s\d{1,2}|  # २०८१ असार १५
\d+%?                       # 5%, १००
)
'''

import re

def get_sentence_vector(sentence, embeddings):
    # Tokenize with stemmer
    tokens = [nepali_stemmer(word) for word in sentence.split()]
    
    # Check for numerical patterns
    has_numerical = bool(re.search(NUMERICAL_PATTERN, sentence, re.VERBOSE))
    
    vectors = []
    for word in tokens:
        if word in embeddings:
            # Apply financial boost if word is in FINANCIAL_BOOST
            boost = FINANCIAL_BOOST.get(word, 1.0)
            vectors.append(embeddings[word] * boost)
    
    if not vectors:
        return np.zeros(100)
    
    avg_vector = np.mean(vectors, axis=0)
    
    # Boost numerical sentences by 25%
    if has_numerical:
        avg_vector *= 1.25
        
    return avg_vector
import networkx as nx

def textrank(sentences, embeddings):
    if not sentences:
        return []
    
    # Dynamic length calculation (keep your original logic)
    num_sentences = len(sentences)
    top_n = max(2, min(10, round(num_sentences * 0.3)))

    # Vectorization (unchanged)
    sentence_vectors = [get_sentence_vector(sent, embeddings) for sent in sentences]
    
    try:
        sentence_vectors = np.array(sentence_vectors)
    except ValueError:
        return sentences[:top_n]

    # ===== Critical PageRank Fixes =====
    similarity_matrix = cosine_similarity(sentence_vectors)
    
    # 1. Remove self-similarities to avoid loops
    np.fill_diagonal(similarity_matrix, 0)
    
    # 2. Add epsilon to ensure connectivity
    similarity_matrix += 1e-8  # Prevents dangling nodes
    
    # 3. Create graph with normalized weights
    nx_graph = nx.from_numpy_array(similarity_matrix)
    
    # 4. Optimized PageRank parameters for convergence
    scores = nx.pagerank(
        nx_graph, 
        alpha=0.85,       # Damping factor (standard)
        max_iter=4000,    # Increased iterations
        tol=1e-12,        # Lower tolerance
    )
    
    # Rank sentences using PageRank scores
    ranked_indices = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [sentences[i] for i in ranked_indices[:top_n]]
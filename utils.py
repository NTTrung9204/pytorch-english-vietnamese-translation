import torch
from collections import Counter
import string

def build_vocab(file_path, max_len=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    all_words = []
    for sentence in sentences:
        sentence = sentence.strip().lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        all_words.extend(sentence.split())

    word_counts = Counter(all_words)
    
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(max_len), 1)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = len(vocab)
    return vocab

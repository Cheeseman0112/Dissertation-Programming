import numpy as np
from collections import Counter

#Build vocabulary based on most frequent words
def buildVocabulary(texts, maxWords = 3000):
    wordCounts = Counter()

    for text in texts:
        words = text.split()
        wordCounts.update(words)

    mostCommon = wordCounts.most_common(maxWords)

    #Word to index mapping
    wordIndex = {word: i+1 for i, (word, _) in enumerate(mostCommon)}

    return wordIndex

#Convert text into sequences of integers
def textsToSequences(texts, wordIndex):
    sequences = []

    for text in texts:
        words = text.split()
        seq = [wordIndex.get(word, 0) for word in words]
        sequences.append(seq)

    return sequences

#Pad or truncate sequences
def padSequences(sequences, maxLen = 50):
    padded = np.zeros((len(sequences), maxLen))

    for i, seq in enumerate(sequences):
        seq = seq[:maxLen]
        padded[i, :len(seq)] = seq

    return padded

#LSTM preprocessing
def preprocessLSTM(X_train, X_test, maxWords = 3000, maxLen = 50):
    wordIndex = buildVocabulary(X_train, maxWords)

    X_train_seq = textsToSequences(X_train, wordIndex)
    X_test_seq = textsToSequences(X_test, wordIndex)

    X_train_pad = padSequences(X_train_seq, maxLen)
    X_test_pad = padSequences(X_test_seq, maxLen)

    return X_train_pad, X_test_pad, wordIndex
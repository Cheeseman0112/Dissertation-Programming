from sklearn.feature_extraction.text import TfidfVectorizer

#Returns a configured TF-IDF vectoriser for SVM
def getTfidfVectorizer():
    return TfidfVectorizer(
        stop_words = "english",
        max_df = 0.75,
        min_df = 5,
        ngram_range = (1, 2)
    )

#Fit TF-IDF on training data and transform both sets
def preprocessSVM(X_train, X_test):
    vectorizer = getTfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer
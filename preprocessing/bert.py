from transformers import DistilBertTokenizer

def getTokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#Tokenisation for DistilBERT
def preprocessBERT(X_train, X_test, maxLen = 512):
    tokenizer = getTokenizer()

    trainEncodings = tokenizer(
        list(X_train),
        truncation = True,
        padding = True,
        max_length = maxLen
    )

    testEncodings = tokenizer(
        list(X_test),
        truncation = True,
        padding = True,
        max_length = maxLen
    )

    return trainEncodings, testEncodings, tokenizer
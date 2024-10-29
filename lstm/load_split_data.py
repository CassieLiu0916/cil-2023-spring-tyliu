import numpy as np

def load_tweets(filenames):
    tweets = []
    labels = []

    for label, filename in enumerate(filenames):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.rstrip())
                labels.append(label)

    tweets = np.array(tweets)
    labels = np.array(labels)

    return tweets, labels

def load_test_tweets(filename):
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip().lstrip('0123456789,'))

    tweets = np.array(tweets)

    return tweets

def train_val_split(tweets, val_size=0.1):
    np.random.seed(1)

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int((1-val_size) * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    return train_indices, val_indices
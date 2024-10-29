import numpy as np
import re

def remove_duplicates(tweets, labels):
    unique_tweets, unique_indices = np.unique(tweets, return_index=True)
    unique_labels = labels[unique_indices]

    return unique_tweets, unique_labels

def glove_preprocess(tweet):
    special_tokens = ['<user>', '<url>', '<number>', '<hashtag>']
    contractions = tuple(('\'s', 'n\'t', '\'m', '\'re', '\'ll', '\'ve', '\'d', '\'n'))

    tweet = re.sub(r'\'+', '\'', tweet)
    tweet = re.sub(r'\.+', '.', tweet)
    tweet = ['<number>' if re.findall(r'\d+', word) else word for word in tweet.split()]
    tweet = ['<hashtag>' if word.startswith('#') else word for word in tweet]
    for word in tweet:
        if '<' in word or '>' in word:
            if word not in special_tokens:
                tweet.remove(word)
    
    # Expand contractions
    expanded_tweet = []
    for word in tweet:
        if word.endswith(contractions):
            if word.endswith('n\'t'):
                expanded = [word[:-3], 'n\'t']
            else:
                expanded = word.split('\'')
                expanded[1] = '\'' + expanded[1]
            expanded_tweet += expanded
        else:
            expanded_tweet.append(word)

    tweet = ' '.join(expanded_tweet)
    
    return tweet
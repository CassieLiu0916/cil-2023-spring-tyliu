from datetime import datetime
import json
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LSTMTweetDataset
from tokenizer import GloveTokenizer
from model import LSTMClassifier
from trainer import LSTMTrainer
from load_split_data import load_tweets, load_test_tweets, train_val_split
from preprocess import remove_duplicates, glove_preprocess
from save_results import save_train_results, save_test_results
from explain import explain_lime

def main():
    with open('config.json') as f:
        config = json.load(f)

    train_filenames = ['../twitter-datasets/train_neg.txt', '../twitter-datasets/train_pos.txt']
    test_filename = '../twitter-datasets/test_data.txt'
    tweets, labels = load_tweets(train_filenames)
    test_tweets = load_test_tweets(test_filename)
    print(f'Loaded {len(tweets)} training tweets and {len(test_tweets)} test tweets.')

    if config['remove_duplicates'] == True:
        tweets, labels = remove_duplicates(tweets, labels)
        print(f'Removed duplicate tweets.')
    if config['preprocess'] == True:
        tweets = np.array([glove_preprocess(tweet) for tweet in tweets])
        test_tweets = np.array([glove_preprocess(tweet) for tweet in test_tweets])
        print(f'Preprocessed tweets.')

    tokenizer = GloveTokenizer()
    tokenized_tweets = tokenizer.tokenize(tweets)
    tokenized_test_tweets = tokenizer.tokenize(test_tweets)
    print(f'Tokenized tweets.')

    train_indices, val_indices = train_val_split(tweets, val_size=config['val_size'])

    tokenized_train_tweets = LSTMTweetDataset(tokenized_tweets[train_indices], labels[train_indices])
    tokenized_val_tweets = LSTMTweetDataset(tokenized_tweets[val_indices], labels[val_indices])

    train_loader = DataLoader(tokenized_train_tweets, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(tokenized_val_tweets, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(tokenized_test_tweets, batch_size=config['batch_size'], shuffle=False)

    model = LSTMClassifier(config)
    if config['model_path'] != '':
        model.load_state_dict(torch.load(config['model_path']))
    print(f'Created LSTM model.')

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    trainer = LSTMTrainer(config, model, device)
    if config['model_path'] == '':
        train_loss, train_accuracy, val_loss, val_accuracy = trainer.train(train_loader, val_loader)
        print(f'Finished training.')

    predictions = trainer.predict(test_loader)
    print(f'Finished predicting.')

    folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir(folder)

    with open(f'{folder}/config.json', 'w') as f:
        json.dump(config, f)
    torch.save(trainer.model.state_dict(), f'{folder}/model.pt')
    if config['model_path'] == '':
        save_train_results(folder, train_loss, train_accuracy, val_loss, val_accuracy)
    save_test_results(folder, predictions)

    # Explain the model's prediction for a test tweet
    explain_lime(folder, trainer.model, test_tweets[config['explain_index']], device)

if __name__ == '__main__':
    main() 
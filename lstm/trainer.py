import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

class LSTMTrainer:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def train(self, train_loader, val_loader):
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []

        if self.device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
        
        sigmoid = nn.Sigmoid()

        for epoch in range(self.config['epochs']):
            self.model.train()

            total_train_loss = 0
            total_train_correct = 0
            total_train = 0

            for tweets, labels in tqdm(train_loader):
                tweets = tweets.to(self.device)
                labels = labels.to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad()

                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = self.model(tweets).squeeze()
                        loss = self.criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(tweets).squeeze()
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                total_train_loss += loss.item()
                outputs = sigmoid(outputs)
                total_train_correct += ((outputs > 0.5) == labels).sum().item()
                total_train += len(labels)

            print(f'Epoch {epoch}:')
            print(f'Training Loss: {total_train_loss / total_train:.05f}, Accuracy: {total_train_correct / total_train:.05f}')
            train_loss.append(total_train_loss / total_train)
            train_accuracy.append(total_train_correct / total_train)

            self.model.eval()

            total_val_loss = 0
            total_val_correct = 0
            total_val = 0

            with torch.no_grad():
                for tweets, labels in val_loader:
                    tweets = tweets.to(self.device)
                    labels = labels.to(self.device, dtype=torch.float32)

                    outputs = self.model(tweets).squeeze()
                    loss = self.criterion(outputs, labels)
                    
                    total_val_loss += loss.item()
                    outputs = sigmoid(outputs)
                    total_val_correct += ((outputs > 0.5) == labels).sum().item()
                    total_val += len(labels)

            print(f'Validation Loss: {total_val_loss / total_val:.05f}, Accuracy: {total_val_correct / total_val:.05f}')
            val_loss.append(total_val_loss / total_val)
            val_accuracy.append(total_val_correct / total_val)

        return train_loss, train_accuracy, val_loss, val_accuracy
        
    def predict(self, test_loader):
        self.model.eval()

        predicions = []
        sigmoid = nn.Sigmoid()

        with torch.no_grad():
            for tweets in test_loader:
                tweets = tweets.to(self.device)
                outputs = self.model(tweets).squeeze()
                outputs = sigmoid(outputs)
                labels = (outputs > 0.5).cpu().numpy().astype(int)
                predicions = predicions + labels.tolist()

        return predicions
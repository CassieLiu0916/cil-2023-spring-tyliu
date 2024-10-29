import pandas as pd

def save_train_results(folder, train_loss, train_accuracy, val_loss, val_accuracy):
    df = pd.DataFrame({
        'epoch': list(range(1, len(train_loss)+1)),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })
    df.to_csv(f'{folder}/train_results.csv', index=False)

def save_test_results(folder, predictions):
    df = pd.DataFrame({
        'Id': list(range(1, len(predictions)+1)),
        'Prediction': predictions
    })
    df['Prediction'] = df['Prediction'].map({0: -1, 1: 1})
    df.to_csv(f'{folder}/test_results.csv', index=False)
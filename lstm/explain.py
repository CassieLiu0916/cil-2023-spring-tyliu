import numpy as np
from lime.lime_text import LimeTextExplainer

from tokenizer import GloveTokenizer

def explain_lime(folder, model, tweets, device):
    model.eval()

    def predict_proba(tweets):
        if isinstance(tweets, str):
            tweets = [tweets]
        tokenizer = GloveTokenizer()
        tweets = tokenizer.tokenize(tweets).to(device)
        output = model(tweets).squeeze().detach().cpu().numpy()
        probs = np.vstack([1-output, output]).T
 
        return probs

    class_names = ['Negative', 'Positive']
    lime_explainer = LimeTextExplainer(class_names=class_names)
    exp = lime_explainer.explain_instance(tweets, predict_proba, num_features=5)
    exp.save_to_file(f'{folder}/lime.html')


from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical

def calculate_metrics(preds, targets):
    accuracy = (preds == targets).mean() * 100.
    quadraditic_weighted_kappa = cohen_kappa_score(preds, targets, weights='quadratic')
    return accuracy, quadraditic_weighted_kappa



def model_checkpoint(model, best_metric, acctualy_metric, path):
    if acctualy_metric > best_metric:
        print(f"Salvando o melhor modelo... {best_metric} -> {acctualy_metric}")
        torch.save(model.state_dict(), path)
        best_metric = acctualy_metric

    return best_metric

def calculate_entropy(predictions=None, logits=None):
    if predictions:
        return Categorical(probs=predictions).entropy()
    return Categorical(logits=logits).entropy()


def compute_image_entropies(model, dataloader):
    model.eval()
    entropies = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            entropy = calculate_entropy(probs)
            entropies.extend(entropy.cpu().numpy())

    return np.array(entropies)


def remove_high_entropy_images(images, entropies, threshold=0.9):
    high_entropy_indices = np.where(entropies >= np.percentile(entropies, 100 - threshold * 100))[0]
    return [img for i, img in enumerate(images) if i not in high_entropy_indices]

from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from tqdm import tqdm

def calculate_metrics(preds, targets):
    accuracy = (preds == targets).mean() * 100.
    kappa = cohen_kappa_score(preds, targets, weights='quadratic')
    return accuracy, kappa



def model_checkpoint(model, best_metric, acctualy_metric, path):
    print(f"Salvando o melhor modelo... {best_metric} -> {acctualy_metric}")
    torch.save(model.state_dict(), path)


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


def evaluation(model, dataloader, device):
    model.eval()
    model.to(device)

    bar_progress = tqdm(dataloader)

    all_logits = []
    predicts = []
    targets = []
    imgs = []

    with torch.no_grad():
        for index, (batch_data, batch_targets, img_ids) in enumerate(bar_progress):
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)
            imgs.extend(img_ids)

            outputs = model(batch_data)

            prediction = outputs.sigmoid()
            prediction = prediction.sum(1).detach().round()

            all_logits.append(outputs)
            predicts.append(prediction)
            targets.append(batch_targets.sum(1))

    predicts = torch.cat(predicts).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    accuracy, kappa = calculate_metrics(predicts, targets)

    return {
        "val_acc": accuracy,
        "kappa": kappa,
    }, (predicts, targets, imgs)
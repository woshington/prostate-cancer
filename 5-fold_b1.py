import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as efficientnet_model
import albumentations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
import skimage
from torch.nn import Identity
from torchvision import models


data_dir = '../dataset'
images_dir = os.path.join(data_dir, 'tiles') 

dataframe_train = pd.read_csv(f"data/train_5fold.csv")
df_test = pd.read_csv(f"data/test.csv")

n_folds = 5 
seed = 42
shuffle = True

batch_size = 2
num_workers = 4
output_classes = 5
init_lr = 3e-4
warmup_factor = 10
loss_function = nn.BCEWithLogitsLoss()

warmup_epochs = 1

n_epochs = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class EfficientNet(nn.Module):
    def __init__(self, output_dimensions):
        super(EfficientNet, self).__init__()

        self.efficient_net = efficientnet_model.EfficientNet.from_pretrained("efficientnet-b1")
        self.efficient_net.load_state_dict(torch.load("pre-trained-models/efficientnet-b1-dbc7070a.pth", weights_only=True))
        self.fully_connected = nn.Linear(self.efficient_net._fc.in_features, output_dimensions)
        self.efficient_net._fc = Identity()


    def extract(self, inputs):
        return self.efficient_net(inputs)

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.fully_connected(x)

        return x
    

class PandasDataset(Dataset):
    def __init__(self, root_dir, dataframe, transforms=None):
        self.root_dir = root_dir 
        self.dataframe = dataframe
        self.transforms = transforms

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img_id = row.image_id.strip()
        
        file_path = os.path.join(self.root_dir, f'{img_id}.jpg')
        tile_image = skimage.io.imread(file_path)

        if self.transforms is not None:
            tile_image = self.transforms(image=tile_image)['image'] 

        tile_image = tile_image.astype(np.float32) / 255.0 
        tile_image = np.transpose(tile_image, (2, 0, 1))

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.

        return torch.tensor(tile_image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# Albumentations
transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5)    
])

def calculate_metrics(preds, targets, dataframe_valid):
    accuracy = (preds == targets).mean() * 100. # Calcula a acurácia em porcentagem
    quadraditic_weighted_kappa = cohen_kappa_score(preds, targets, weights='quadratic') # Calcula o kappa quadrático ponderado dos dados em geral

    quadraditic_weighted_kappa_karolinska = cohen_kappa_score(preds[dataframe_valid['data_provider'] == 'karolinska'], dataframe_valid[dataframe_valid['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    quadraditic_weighted_kappa_radboud = cohen_kappa_score(preds[dataframe_valid['data_provider'] == 'radboud'], dataframe_valid[dataframe_valid['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')

    return accuracy, quadraditic_weighted_kappa, quadraditic_weighted_kappa_karolinska, quadraditic_weighted_kappa_radboud


def training_step(model, dataset_loader, optimizer, epoch):
    """
        Função que realiza uma etapa de treinamento do modelo

        Parâmetros:
            model: Modelo a ser treinado
            dataset_loader: DataLoader do PyTorch
            optimizer: Otimizador a ser utilizado

        Retorna:
            train_loss: Lista com o valor do loss de treinamento
    """
    model.train()
    train_loss = []

    bar_progress = tqdm(dataset_loader)

    for index, (batch_data, batch_targets) in enumerate(bar_progress):
        batch_data, batch_targets = batch_data.to(device), batch_targets.to(device)

        optimizer.zero_grad()  # Zera os gradientes
        
        logits = model(batch_data)   # Classificação do lote de dados / Retorna o valores brutos dos neurônios (logits)
        
        loss = loss_function(logits, batch_targets)  # Calcula a perda de treino

        loss.backward()  # Aplica o backpropagation

        optimizer.step()  # Atualiza os pesos do modelo com base nos gradientes calculados durante o backpropagation

        loss_np = loss.detach().cpu().numpy() # Converte o loss para numpy e remove a dependência do grafo computacional (detach)
        train_loss.append(loss_np)
        #Calcula a perda média suavizada, considerando as últimas 100 perdas. Isso ajuda a monitorar a evolução da perda ao longo 
        # do tempo sem ser influenciado por variações bruscas.
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        
        print(f'epoch: {epoch} batch: {(index+1)} of {len(bar_progress)} loss: {loss_np}, smooth loss: {smooth_loss}')
        bar_progress.set_description('loss: %.5f, smooth loss: %.5f' % (loss_np, smooth_loss)) # Atualiza a barra de progresso
        
    return train_loss # Retorna a perda de treino


def validation_step(model, dataset_loader, dataframe_valid, get_logits = False):
    model.eval() # Define o modelo em modo de avaliação

    validation_loss = [] # Armazena o loss de validação
    all_logits = [] # Armazena os valores brutos dos neurônios (logits)
    preds = []
    targets = []

    bar_progress = tqdm(dataset_loader) # Barra de progresso

    with torch.no_grad(): # Desativa o cálculo de gradientes
        for index, (batch_data, batch_targets) in enumerate(bar_progress): # Itera sobre o DataLoader 
            batch_data, batch_targets = batch_data.to(device), batch_targets.to(device) # Move os dados para o device

            logits = model(batch_data) # Classificação do lote de dados / Retorna o valores brutos dos neurônios (logits)   

            loss = loss_function(logits, batch_targets) # Calcula a perda entre os logits previstos e os targets

            prediction = logits.sigmoid() # Aplica a função sigmoid para converter os logits em probabilidades
            prediction = prediction.sum(1).detach().round() #  Realiza a soma do valores de saída, removendo a dependência do grafo computacional (detach)
                                                            # e arredonda o valor para o inteiro mais próximo
        
            all_logits.append(logits) # Salva os logits na lista
            preds.append(prediction) # Salva as predições na lista
            targets.append(batch_targets.sum(1)) # realiza a soma dos targets para obter o valor do isup_grade (rotulo real) e salva na lista

            validation_loss.append(loss.detach().cpu().numpy()) # Salva o loss de validação

        validation_loss = np.mean(validation_loss) # Calcula a perda média de validação durante a época

    # Concatena os logits, predições e targets
    all_logits = torch.cat(all_logits).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    # Após inferir todos os dados de validação, calcula as métricas, inclusive o kappa quadrático ponderado para cada data_provider
    accuracy, quadraditic_weighted_kappa, quadraditic_weighted_kappa_karolinska, quadraditic_weighted_kappa_radboud = calculate_metrics(preds, targets, dataframe_valid)

    # Retorna as métricas se get_logits for False, caso contrário, retorna os logits (valores brutos dos neurônios)
    if not get_logits:
        return {
            "val_loss":validation_loss, 
            "val_acc":accuracy, 
            "quadraditic_weighted_kappa":quadraditic_weighted_kappa, 
            "quadraditic_weighted_kappa_karolinska":quadraditic_weighted_kappa_karolinska, 
            "quadraditic_weighted_kappa_radboud":quadraditic_weighted_kappa_radboud
        }
    else:
        all_logits


def model_checkpoint(model, best_metric, acctualy_metric, path):
    """
        Função que salva o modelo

        Parâmetros:
            model: Modelo a ser salvo
            best_metric: Melhor métrica
            acctualy_metric: Métrica atual
            path: Caminho para salvar o modelo
    """

    if acctualy_metric > best_metric:
        print(f"Salvando o melhor modelo... {best_metric} -> {acctualy_metric}")
        torch.save(model.state_dict(), path)
        best_metric = acctualy_metric

    return best_metric


def train_model(fold, model, epochs, optimizer, scheduler, train_dataloader, valid_dataloader, valid_dataframe, path_to_save_model):
    best_metric_criteria = 0. # Critério de melhor métrica
    
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}\n')

        train_loss = training_step(model, train_dataloader, optimizer, epoch) # Realiza a etapa de treinamento
        metrics = validation_step(model, valid_dataloader, valid_dataframe) # Realiza a etapa de validação

        log_epoch = f'fold: {fold} | epoch: {epoch} | lr: {optimizer.param_groups[0]["lr"]:.7f} | Train loss: {np.mean(train_loss)} | Validation loss: {metrics["val_loss"]} | Validation accuracy: {metrics["val_acc"]} | QWKappa: {metrics["quadraditic_weighted_kappa"]} | QWKappa Karolinska: {metrics["quadraditic_weighted_kappa_karolinska"]} | QWKappa Radboud: {metrics["quadraditic_weighted_kappa_radboud"]}'
        with open('logs/history/folds_b1.txt', 'a') as f:
            f.write(log_epoch + '\n')
        
        # Salva o modelo se a métrica atual for melhor que a melhor métrica / Atualmente a métrica é o kappa quadrático ponderado
        best_metric_criteria = model_checkpoint(model, best_metric_criteria, metrics["quadraditic_weighted_kappa"], path_to_save_model)

        scheduler.step() # Atualiza o scheduler

fold = 3
print(f"Iniciando treino do fold: {fold}")

train_indexes = np.where((dataframe_train['fold'] != fold))[0] # Pega os índices de treino / Todos os índices que não correspondem ao fold atual são de treino
valid_indexex = np.where((dataframe_train['fold'] == fold))[0] # Pega os índices de validação

dataframe_train_fold = dataframe_train.loc[train_indexes]
dataframe_valid_fold = dataframe_train.loc[valid_indexex]

dataset_train = PandasDataset(images_dir, dataframe_train_fold, transforms = transforms_train) # Instancia o dataset de treino
dataset_valid = PandasDataset(images_dir, dataframe_valid_fold, transforms = None) # Instancia o dataset de validação

print(f"train: {len(dataset_train)} images | validation: {len(dataset_valid)} images")

# Inicia os dataloaders de treino e validação
train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler = RandomSampler(dataset_train), num_workers = num_workers)
valid_dataloader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, sampler = SequentialSampler(dataset_valid), num_workers = num_workers)

model = EfficientNet(output_dimensions = output_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = init_lr / warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epochs)
scheduler = GradualWarmupScheduler(optimizer, multiplier = warmup_factor, total_epoch = warmup_epochs, after_scheduler=scheduler_cosine)

save_path = f'pre-trained-models/folds/fold_{fold}_b1.pth'

train_model(fold, model, n_epochs, optimizer, scheduler, train_dataloader, valid_dataloader, dataframe_valid_fold, save_path)

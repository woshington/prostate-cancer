# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import model as efficientnet_model
import albumentations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from skimage import io as skio

# %%
data_dir = 'dataset' # Diretório raiz dos dados
images_dir = os.path.join(data_dir, 'tiles') # Path para o diretório das imagens

df_train = pd.read_csv("data/train.csv")
df_val = pd.read_csv("data/val.csv")
df_test = pd.read_csv("data/test.csv")
df_val = df_val.drop(columns=["Unnamed: 0"])
df_test = df_test.drop(columns=["Unnamed: 0"])
df_train_val = pd.concat([df_train, df_val])

df_pen_mask = pd.read_csv("data/without_pen_mask.csv")

df_filtered = df_train_val[~df_train_val['image_id'].isin(df_pen_mask['image_id'])]



n_folds = 5 # Número de folds da validação cruzada
seed = 42 # Semente aleatória
shuffle = True # Embaralha os dados

batch_size = 1 # Tamanho do batch
num_workers = 4 #N'úmero de processos paralelos que carregam os dados
output_classes = 5 # Número de classes
init_lr = 3e-4 # Taxa de aprendizado inicial
warmup_factor = 10 #Fator de aquecimento para aumentar gradualmente a taxa de aprendizado no início do treinamento.
loss_function = nn.BCEWithLogitsLoss() # Função de perda

warmup_epochs = 1 #Número de épocas de warmup, durante as quais a taxa de aprendizado aumenta progressivamente.

n_epochs = 30 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataframe_train = df_filtered.reset_index(drop=True)
dataframe_train.columns = dataframe_train.columns.str.strip() # Remove espaços em branco


stratified_k_fold = StratifiedKFold(n_folds, shuffle = shuffle, random_state=seed)

dataframe_train['fold'] = -1 

for i, (train_indexes, valid_indexes) in enumerate(stratified_k_fold.split(dataframe_train, dataframe_train['isup_grade'])):
    dataframe_train.loc[valid_indexes, 'fold'] = i
    
dataframe_train.head()


# %%
class EfficientNet(nn.Module):
    """
        Classe que implementa a arquitetura EfficientNet

        Parâmetros:
            backbone: str
                Nome do modelo de EfficientNet a ser utilizado
            output_dimensions: int
                Número de neuronios na camada de saída
    """
    def __init__(self, output_dimensions):
        super().__init__()
        self.efficient_net = efficientnet_model.EfficientNet.from_pretrained("efficientnet-b0")
        self.efficient_net.load_state_dict(
            torch.load(
                "pre-trained-models/efficientnet-b0-08094119.pth",
                weights_only=True
            )
        )
        self.fully_connected = nn.Linear(self.efficient_net._fc.in_features, output_dimensions)
        self.efficient_net._fc = nn.Identity()

    def extract(self, inputs):
        return self.efficient_net(inputs)

    def forward(self, inputs):
        x = self.extract(inputs)
        x = self.fully_connected(x)

        return x

# %%
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
        tile_image = skio.imread(file_path)
        
        # Substituir pixels brancos por pretos
        white = np.array([255, 255, 255])
        black = np.array([0, 0, 0])
        white_pixels = np.all(tile_image == white, axis=-1)  # Identifica pixels brancos
        tile_image[white_pixels] = black  # Substitui os pixels brancos por preto


        if self.transforms is not None:
            tile_image = self.transforms(image=tile_image)['image'] 

        tile_image = tile_image.astype(np.float32) / 255.0 
        tile_image = np.transpose(tile_image, (2, 0, 1))

        label = np.zeros(5).astype(np.float32)
        label[:row.isup_grade] = 1.
        
        return torch.tensor(tile_image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# %%
# Albumentations
transforms_train = albumentations.Compose([
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])

# %%
def calculate_metrics(preds, targets, df):
    accuracy = (preds == targets).mean() * 100. # Calcula a acurácia em porcentagem
    quadraditic_weighted_kappa = cohen_kappa_score(preds, targets, weights='quadratic') # Calcula o kappa quadrático ponderado dos dados em geral

    quadraditic_weighted_kappa_karolinska = cohen_kappa_score(preds[df['data_provider'] == 'karolinska'], df[df['data_provider'] == 'karolinska'].isup_grade.values, weights='quadratic')
    quadraditic_weighted_kappa_radboud = cohen_kappa_score(preds[df['data_provider'] == 'radboud'], df[df['data_provider'] == 'radboud'].isup_grade.values, weights='quadratic')

    return accuracy, quadraditic_weighted_kappa, quadraditic_weighted_kappa_karolinska, quadraditic_weighted_kappa_radboud



# %%
dataset_test = PandasDataset(images_dir, df_test, transforms = None)
dataloader_test = DataLoader(dataset_test, batch_size=5, shuffle=False)

# %%
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


def train_model(model, epochs, optimizer, scheduler, train_dataloader, valid_dataloader, valid_dataframe, path_to_save_model):
    best_metric_criteria = 0. # Critério de melhor métrica
    
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')

        train_loss = training_step(model, train_dataloader, optimizer, epoch) # Realiza a etapa de treinamento
        metrics = validation_step(model, valid_dataloader, valid_dataframe) # Realiza a etapa de validação

        log_epoch = f'lr: {optimizer.param_groups[0]["lr"]:.7f} | Train loss: {np.mean(train_loss)} | Validation loss: {metrics["val_loss"]} | Validation accuracy: {metrics["val_acc"]} | QWKappa: {metrics["quadraditic_weighted_kappa"]} | QWKappa Karolinska: {metrics["quadraditic_weighted_kappa_karolinska"]} | QWKappa Radboud: {metrics["quadraditic_weighted_kappa_radboud"]}'
        with open('logs/white_black.txt', 'a') as f:
            f.write(log_epoch + '\n')
        
        # Salva o modelo se a métrica atual for melhor que a melhor métrica / Atualmente a métrica é o kappa quadrático ponderado
        best_metric_criteria = model_checkpoint(model, best_metric_criteria, metrics["quadraditic_weighted_kappa"], path_to_save_model)

        scheduler.step() # Atualiza o scheduler

# %%
for fold in range(n_folds):
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

    model = EfficientNet(output_dimensions = output_classes) # Instancia o modelo EfficientNet pré-treinado
    model = model.to(device) # Move o modelo para o device (GPU se disponível)

    optimizer = optim.Adam(model.parameters(), lr = init_lr / warmup_factor) # Inicializa o otimizador Adam reduzindo a taxa de aprendizado inicial durante as primeiras iterações para evitar grandes ajustes logo no inicio
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epochs) # Ajusta a taxa de aprendizado de acordo com a função cosseno / vai reduzindo de forma suave, de acordo com a função cosseno.
                                                                                                    # Os ajustes do scheduler são realizados somente após a fase de warmup
    scheduler = GradualWarmupScheduler(optimizer, multiplier = warmup_factor, total_epoch = warmup_epochs, after_scheduler=scheduler_cosine) # Ajusta a taxa de aprendizado gradualmente durante a fase de warmup, depois utiliza o scheduler_cosine

    save_path = f'pre-trained-models/white_black/fold_{fold}.pth'

    train_model(model, n_epochs, optimizer, scheduler, train_dataloader, valid_dataloader, dataframe_valid_fold, save_path)


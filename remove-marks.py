from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt

def calculate_histogram(chip):
    """
    Calcula o histograma para cada canal de uma imagem RGB.
    """
    histograms = []
    # # Extrair histograma de cada canal
    hist_r, _ = np.histogram(chip[..., 0].ravel(), bins=256, range=[0, 256])
    hist_g, _ = np.histogram(chip[..., 1].ravel(), bins=256, range=[0, 256])
    hist_b, _ = np.histogram(chip[..., 2].ravel(), bins=256, range=[0, 256])

    histograms.append(hist_r)
    histograms.append(hist_g)
    histograms.append(hist_b)
    return histograms

def calculate_channel_sums(image):
    """
    Calcula a soma das intensidades para cada canal da imagem RGB.

    Args:
        image (np.ndarray): Imagem RGB.

    Returns:
        tuple: Somas das intensidades para os canais (red_sum, green_sum, blue_sum).
    """
    red_sum = np.sum(image[:, :, 0])  # Canal vermelho
    green_sum = np.sum(image[:, :, 1])  # Canal verde
    blue_sum = np.sum(image[:, :, 2])  # Canal azul
    return red_sum, green_sum, blue_sum

def analyze_histograms(image, threshold=1.5):
    """
    Analisa os histogramas para detectar predominância de um canal (verde ou azul).

    Args:
        histograms (list): Lista com os histogramas dos canais RGB.
        threshold (float): Fator pelo qual um canal deve exceder os outros para ser considerado predominante.

    Returns:
        bool: True se a imagem deve ser removida (predominância verde ou azul), False caso contrário.
    """

    red_sum, green_sum, blue_sum = calculate_channel_sums(image)

    green_dominance = green_sum > threshold * red_sum
    blue_dominance = blue_sum > threshold * red_sum

    if green_dominance or blue_dominance:
        return True

    return False

def extract_features(image, chip_size=(256, 256), overlap=0, threshold=1.1, output_dir='imagens/processadas/com_marcas'):
    """
    Extrai features de uma imagem usando chips com tamanho fixo.

    Args:
        image (str ou np.ndarray): Caminho para a imagem ou imagem carregada como NumPy array.
        chip_size (tuple): Tamanho do chip (altura, largura).
        overlap (int): Quantidade de sobreposição entre chips.

    Returns:
        np.ndarray: Matriz de features com shape (n_chips, n_features).
    """
    if isinstance(image, str):
        image = np.array(Image.open(image))

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    height, width = image.shape[:2]
    chip_h, chip_w = chip_size

    images = []

    for y in range(0, height, chip_h - overlap):
        for x in range(0, width, chip_w - overlap):
            chip = image[y:y + chip_h, x:x + chip_w]

            # Preencher bordas com zeros, se necessário
            if chip.shape[0] < chip_h or chip.shape[1] < chip_w:
                padded_chip = np.zeros((chip_h, chip_w, image.shape[2]), dtype=image.dtype)
                padded_chip[:chip.shape[0], :chip.shape[1]] = chip
                chip = padded_chip

            if analyze_histograms(chip, threshold):
                image[y:y + chip_h, x:x + chip_w] = 255

    new_image = Image.fromarray(image)
    os.makedirs(output_dir, exist_ok=True)
    new_filename = os.path.basename(image_path)
    new_image.save(f'{output_dir}/{new_filename}')

if __name__ == '__main__':
    # path_images = '/Users/Oseas/Documents/test_oxito/imagens/com marcas'
    # for image in os.listdir(path_images):
    #     if image.endswith('.jpg'):
    #         image_path = f'/Users/Oseas/Documents/test_oxito/imagens/com marcas/{image}'
    image_path = "../dataset/tiles/c4b1a10db8b0cdece7a1498b2fcbda7f.jpg"
    extract_features(image=image_path, chip_size=(16, 16), overlap=0, threshold=1)



import streamlit as st
import cv2
import numpy as np
from PIL import Image


import cv2
import numpy as np
import torch
import albumentations as albu
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

# Define the classes and image dimensions
CLASSES = ["фон", "волосы", "кожа"]
INFER_WIDTH = 256
INFER_HEIGHT = 256

# Define the normalization statistics for ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define the device to use for computations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the JIT model
best_model = torch.jit.load('models/best_model_new.pt', map_location=DEVICE)

# Define the augmentations
def get_validation_augmentation():
    test_transform = [
        albu.LongestMaxSize(max_size=INFER_HEIGHT, always_apply=True),
        albu.PadIfNeeded(min_height=INFER_HEIGHT, min_width=INFER_WIDTH, border_mode=0, always_apply=True),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return albu.Compose(test_transform)


def infer_image(image):
    # Инференс на изображении Unet
    original_height, original_width, _ = image.shape

    # Apply the augmentations
    augmentation = get_validation_augmentation()
    augmented = augmentation(image=image)
    image_transformed = augmented['image']

    # Convert the image to a PyTorch tensor and move it to the device
    x_tensor = torch.from_numpy(image_transformed).to(DEVICE).unsqueeze(0).permute(0, 3, 1, 2).float()

    # Run the image through the model
    best_model.eval()
    with torch.no_grad():
        pr_mask = best_model(x_tensor)

    # Convert the output to a numpy array and remove the batch dimension
    pr_mask = pr_mask.squeeze().cpu().detach().numpy()

    # Get the class with the highest probability for each pixel
    label_mask = np.argmax(pr_mask, axis=0)

    # Опредяем сколько пикселей появится по бокам от паддингов и далее вырежем их
    if original_height > original_width:
        delta_pixels = int(((original_height-original_width)/2)/original_height * INFER_HEIGHT)
        print('delta_pixels', delta_pixels)
        image_cropped = image_transformed[:, delta_pixels + 1: INFER_WIDTH - delta_pixels - 1]
        mask_cropped = label_mask[:, delta_pixels + 1 : INFER_WIDTH - delta_pixels - 1]
    elif original_height < original_width:
        delta_pixels = int(((original_width-original_height)/2)/original_width * INFER_WIDTH)
        print('delta_pixels', delta_pixels)
        image_cropped = image_transformed[delta_pixels + 1: INFER_HEIGHT - delta_pixels - 1, :]
        mask_cropped = label_mask
    else:
        mask_cropped = label_mask
        image_cropped = image_transformed

    # Resize the mask back to the original image size
    label_mask_real_size = cv2.resize(
        mask_cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    return label_mask_real_size


# Функция для обновления изображения на основе выбранных пользователем значений
def adjust_hsv(image, h_adjust, s_adjust, v_adjust):
    # Преобразование изображения в HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(image_hsv)
    
    # Применение корректировок
    h = np.clip(h + h_adjust, 0, 179)
    s = np.clip(s + s_adjust, 0, 255)
    v = np.clip(v + v_adjust, 0, 255)
    
    # Объединение каналов HSV обратно в одно изображение
    image_hsv_adjusted = cv2.merge([h, s, v])
    
    # Преобразование изображения обратно в RGB для отображения
    image_rgb_adjusted = cv2.cvtColor(image_hsv_adjusted.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return image_rgb_adjusted


# Функция для отображения изображения
def display_image(image):
    st.image(image, use_column_width=True)

# Функция для загрузки изображения
def upload_image(label):
    uploaded_file = st.file_uploader(label, type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image_data = np.array(Image.open(uploaded_file))
        return image_data
    return None

# Основная функция приложения
def main():
    st.title('Инструмент корректировки изображений')

    # Загрузка изображения
    image = upload_image('Загрузите изображение')

    # Проверка, что изображение загружено
    if image is not None:
        # Выбор значений для корректировки HSV
        h_adjust = st.slider('Корректировка оттенка (H) (-179 до 179)', -179, 179, 0)
        s_adjust = st.slider('Корректировка насыщенности (S) (-255 до 255)', -255, 255, 0)
        v_adjust = st.slider('Корректировка освещения (V) (-255 до 255)', -255, 255, 0)

        # Выбор значения для изменения в маске с помощью выпадающего списка
        mask_value = st.selectbox('Выберите интересующую область', CLASSES)

        # Ищем индекс значения в списке
        index = CLASSES.index(mask_value)

        mask = infer_image(image)

        # Применение корректировок HSV
        adjusted_image = adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index)

        # Отображение исходного изображения и скорректированного изображения в двух столбцах
        col1, col2 = st.columns(2)
        with col1:
            display_image(image)
        with col2:
            display_image(adjusted_image)

if __name__ == '__main__':
    main()

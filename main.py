import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter



#Количество классов - фон + число объектов, которые нужно распознать
CLASSES = 2
#Цвета которыми нужно подсветить объекты
COLORS = ['black', 'violet']
#Размеры входного изображения, которое будет обрабатываться нейронкой
SAMPLE_SIZE = (256, 256)
#Размеры выходного изображения, которое будет сформировано после всех обработок
OUTPUT_SIZE = (1080, 1920)


#Функция загрузки изображения и маски, их последующей обработки
def load_images(image, mask):
    #Считывание изображения и возврат тензора со строкой байтов, представляющей содержимое файла
    image = tf.io.read_file(image)
    #Декодирование содержимого JPEG из тензора байтов
    image = tf.io.decode_jpeg(image)
    #Изменение размера
    image = tf.image.resize(image, OUTPUT_SIZE)
    #Преобразование типа данных изображения во float32, необходимо так как с этим типом работает нейронка
    image = tf.image.convert_image_dtype(image, tf.float32)
    #Нормализация данных для упрощения обучения
    image = image / 255.0
    #Тоже самое проделываем с маской
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask)
    #Преобразование в оттенки серого. Необходимо для уменьшения шума, и сегментации изображения
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)

    #Объявление списка для разбиения классов по нескольким каналам
    masks = []
    #Цикл для разбиения. В списке будут содержаться нули или единицы показывая наличие или отсутствие объекта
    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))

    #Объединение масок вдоль оси `axis=2` в один тензор. Каждый пиксель будет представлен вектором длины `CLASSES`
    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

    return image, masks


#Функция для аугментации данных - искусственное увеличение набора данных
def augmentate_images(image, masks):
    random_crop = tf.random.uniform((), 0.3, 1)
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)

    random_flip = tf.random.uniform((), 0, 1)
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)

    image = tf.image.resize(image, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)

    return image, masks


#Загрузка данных с диска
images = sorted(glob.glob("img/*.jpg"))
masks = sorted(glob.glob("masks/*.png"))
#Формирование набора данных из изображений и масок
images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(masks)
#Объединение для параллельной обработки
dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))
#Загрузка данных в память
dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
#Искусственное увеличение объема данных путем копирования уже имеющихся изображений
dataset = dataset.repeat(60)
#Аугментация всех данных, теперь каждое изображение 'уникально'
dataset = dataset.map(augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

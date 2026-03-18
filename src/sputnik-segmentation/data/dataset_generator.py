#
#  class DatasetGenerator:
#     def __init__(self):
#         pass

import osmnx as ox
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from PIL import Image
import numpy as np
import ee
import requests
from io import BytesIO
import matplotlib.pyplot as plt

ee.Authenticate()
ee.Initialize(project='my-map-segmentation')

width = 1024
height = 1024
tags = {"building": True}


def get_square_coords(lat: float, lon: float, size_m: int) -> tuple:
    # Константа: сколько метров в одном градусе широты
    meters_per_degree = 111320

    # Смещение по широте (latitude) одинаково везде
    delta_lat = size_m / meters_per_degree

    # Смещение по долготе (longitude) зависит от текущей широты
    import math
    delta_lon = size_m / (meters_per_degree * math.cos(math.radians(lat)))

    # Верхняя левая точка (уже дана)
    top_left = (lat, lon)

    # Нижняя правая точка:
    # Идем "вниз" по широте (минус) и "вправо" по долготе (плюс)
    bottom_right = (lat - delta_lat, lon + delta_lon)

    return top_left, bottom_right


# Метод для получения маски по ббоксу
def get_mask(minx: float, miny: float, maxx: float, maxy: float) -> Image.Image:
    gdf = ox.features_from_bbox((minx, miny, maxx, maxy), tags)
    objects = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    bbox = box(minx, miny, maxx, maxy)
    clipped_objects = objects.clip(bbox)
    mask = rasterize(
        [(geom, 1) for geom in clipped_objects.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    img_mask = Image.fromarray(mask * 255)
    return img_mask


def get_photo(minx: float, miny: float, maxx: float, maxy: float):
    bbox = ee.Geometry.BBox(minx, miny, maxx, maxy)

    # 1. Get the raw image
    img = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(bbox) \
        .filterDate('2023-01-01', '2023-12-31') \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() \
        .select(['B4', 'B3', 'B2']) \
        .visualize(min=0, max=3000, gamma=1.4)

    url = img.getThumbURL({
        'dimensions': 1024,  # Максимальный размер одной из сторон в пикселях
        'region': bbox,
        'format': 'png'
    })
    # Загружаем данные по ссылке прямо в переменную
    response = requests.get(url)
    if response.status_code == 200:
        pil_img = Image.open(BytesIO(response.content))
        return pil_img
    else:
        print("Ошибка загрузки:", response.text)
        return None

top_left, bottom_right = get_square_coords(39.00, 51.50, 5000)

mask = get_mask(top_left[0], top_left[1], bottom_right[0], bottom_right[1])
print('маска')
photo = get_photo(top_left[0], top_left[1], bottom_right[0], bottom_right[1])
print('фото')

plt.imshow(photo)


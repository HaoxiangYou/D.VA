import numpy as np
from PIL import Image
import random

def save_checkerboard(path='checkerboard.png', total_length=200, tile_length=1, res=2):
    size = total_length * res
    img = np.zeros((size, size, 3), dtype=np.uint8)
    num_tiles = total_length // tile_length
    for i in range(num_tiles):
        for j in range(num_tiles):
            if (i + j) % 2 == 0:
                img[i*res:(i+1)*res, j*res:(j+1)*res] = 255
    Image.fromarray(img).save(path)

def save_skybox(path='skybox.png', total_length=200, top_rgb=(0.4, 0.6, 0.8), res=2, star_size=1,
                       bottom_rgb=(0, 0, 0), num_stars=1000):
    size = res * total_length
    top = np.array(top_rgb) * 255
    bottom = np.array(bottom_rgb) * 255
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        ratio = y / size
        color = (1 - ratio) * top + ratio * bottom
        img[y, :] = color.astype(np.uint8)
    
    # Add random stars
    half_star_pixel = int(star_size * res/2)
    for _ in range(num_stars):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        img[y-half_star_pixel:y+half_star_pixel, x-half_star_pixel:x+half_star_pixel] = [255, 255, 255]  # white star    
    Image.fromarray(img).save(path)

save_checkerboard()
save_skybox()

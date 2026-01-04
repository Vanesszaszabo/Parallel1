import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

KERNEL_SIZE = 7
NUM_THREADS = 4

def process_block(block, mode=''):
    if mode == 'gauss':
        return cv2.GaussianBlur(block, (KERNEL_SIZE, KERNEL_SIZE), 0)
    elif mode == 'edge':
        return cv2.Canny(block, 100, 200)
    elif mode == 'median':
        return cv2.medianBlur(block, KERNEL_SIZE)
    else:
        raise ValueError("Unknown processing mode!")

def split_image(image, n_parts):
    h = image.shape[0]
    step = h // n_parts
    return [image[i*step:(i+1)*step] for i in range(n_parts)]

def combine_blocks(blocks):
    return np.vstack(blocks)

if __name__ == "__main__":
    total_start = time.time()

    img = cv2.imread("input.png")
    if img is None:
        raise FileNotFoundError("No finded input.jpg")

    blocks = split_image(img, NUM_THREADS)

    start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(lambda b: process_block(b, 'gauss'), blocks))
    end = time.time()

    processed = combine_blocks(results)
    cv2.imwrite("output_gauss.jpg", processed)
    print(f"Gauss-blur: {end - start:.3f} s")

    blocks = split_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), NUM_THREADS)
    start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(lambda b: process_block(b, 'edge'), blocks))
    end = time.time()

    processed = combine_blocks(results)
    cv2.imwrite("output_edge.jpg", processed)
    print(f"Edge: {end - start:.3f} s")

    blocks = split_image(img, NUM_THREADS)
    start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(lambda b: process_block(b, 'median'), blocks))
    end = time.time()
    processed = combine_blocks(results)
    cv2.imwrite("output_median.jpg", processed)
    print(f"Median: {end - start:.3f} s")

    total_end = time.time()
    print(f"Total runtime: {total_end - total_start:.3f} s")

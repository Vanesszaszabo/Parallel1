import cv2
import numpy as np
import time

KERNEL_SIZE = 7
NUM_BLOCKS = 4  # eredetileg NUM_THREADS

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
        raise FileNotFoundError("No finded input.png")

    # --- Gauss Blur ---
    blocks = split_image(img, NUM_BLOCKS)
    start = time.time()
    results = [process_block(b, 'gauss') for b in blocks]
    end = time.time()
    processed = combine_blocks(results)
    cv2.imwrite("output_gauss.jpg", processed)
    print(f"Gauss-blur: {end - start:.3f} s")

    # --- Edge Detection ---
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blocks = split_image(gray_img, NUM_BLOCKS)
    start = time.time()
    results = [process_block(b, 'edge') for b in blocks]
    end = time.time()
    processed = combine_blocks(results)
    cv2.imwrite("output_edge.jpg", processed)
    print(f"Edge: {end - start:.3f} s")

    # --- Median Blur ---
    blocks = split_image(img, NUM_BLOCKS)
    start = time.time()
    results = [process_block(b, 'median') for b in blocks]
    end = time.time()
    processed = combine_blocks(results)
    cv2.imwrite("output_median.jpg", processed)
    print(f"Median-blur: {end - start:.3f} s")

    total_end = time.time()
    print(f"Total runtime: {total_end - total_start:.3f} s")

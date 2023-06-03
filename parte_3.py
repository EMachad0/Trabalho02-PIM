import cv2
import math
import numpy as np

Pwx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Pwy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

Sbx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sby = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

Scx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
Scy = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])


def apply_kernel(img, kernel):
    sz = kernel.shape[0] // 2

    # add mirror board
    img_padded = cv2.copyMakeBorder(img, sz, sz, sz, sz, cv2.BORDER_REFLECT)

    # Create output image
    output_img = np.zeros_like(img_padded, dtype=np.int32)

    for i in range(sz, img_padded.shape[0] - sz):
        for j in range(sz, img_padded.shape[1] - sz):
            neigh = img_padded[i - sz:i + sz + 1, j - sz:j + sz + 1]
            output_img[i, j] = np.sum(neigh * kernel)

    # Remove mirror board
    output_img = output_img[sz:-sz, sz:-sz]

    # Normalize
    output_img = (output_img - np.min(output_img)) / (np.max(output_img) - np.min(output_img)) * 255

    return output_img.astype(np.uint8)


def gradient_magnitude(gx, gy):
    gx = gx.astype(np.int32)
    gy = gy.astype(np.int32)
    magnitude = np.around(np.sqrt(gx ** 2 + gy ** 2))
    magnitude = magnitude / np.max(magnitude) * 255

    return magnitude.astype(np.uint8)


def gradient_direction(gx, gy):
    gx = gx.astype(np.int32)
    gy = gy.astype(np.int32)
    eps = 10 ** (-8)
    direction = np.degrees(np.arctan2(gy, gx + eps))
    direction = (direction - np.min(direction)) / (np.max(direction) - np.min(direction)) * 255

    return direction.astype(np.uint8)


def local_maximum(gx, gy):
    gx = gx.astype(np.int32)
    gy = gy.astype(np.int32)
    magnitude = np.around(np.sqrt(gx ** 2 + gy ** 2))
    eps = 10 ** (-8)
    direction = np.degrees(np.arctan2(gy, gx + eps))

    def get_neighbours(i, j):
        mapping = {
            'A': (i - 1, j - 1),
            'B': (i - 1, j),
            'C': (i - 1, j + 1),
            'D': (i, j - 1),
            'F': (i, j + 1),
            'G': (i + 1, j - 1),
            'H': (i + 1, j),
            'I': (i + 1, j + 1)
        }
        if 22.5 < direction[i, j] <= 67.5:
            return mapping['C'], mapping['G']
        elif 67.5 < direction[i, j] <= 112.5:
            return mapping['B'], mapping['H']
        elif 112.5 < direction[i, j] <= 157.5:
            return mapping['A'], mapping['I']
        elif 157.5 < direction[i, j] <= 180:
            return mapping['D'], mapping['F']
        elif -22.5 < direction[i, j] <= 22.5:
            return mapping['F'], mapping['D']
        elif -67.5 < direction[i, j] <= -22.5:
            return mapping['I'], mapping['A']
        elif -112.5 < direction[i, j] <= -67.5:
            return mapping['H'], mapping['B']
        elif -157.5 < direction[i, j] <= -112.5:
            return mapping['G'], mapping['C']
        elif -180 <= direction[i, j] <= -157.5:
            return mapping['D'], mapping['F']

    output_img = np.zeros_like(magnitude, dtype=np.int32)

    for i in range(1, direction.shape[0] - 1):
        for j in range(1, direction.shape[1] - 1):
            y, x = get_neighbours(i, j)
            yi, yj = y
            xi, xj = x
            if magnitude[i, j] > magnitude[yi, yj] and magnitude[i, j] > magnitude[xi, xj]:
                output_img[i, j] = magnitude[i, j]

    output_img = output_img / np.max(output_img) * 255
    return output_img.astype(np.uint8)


def vertical_components(img):
    pw = apply_kernel(img, Pwx)
    sb = apply_kernel(img, Sbx)
    sc = apply_kernel(img, Scx)

    return pw, sb, sc


def horizontal_components(img):
    pw = apply_kernel(img, Pwy)
    sb = apply_kernel(img, Sby)
    sc = apply_kernel(img, Scy)

    return pw, sb, sc

def median_filter(img):
    output_img = np.zeros_like(img, dtype=np.int32)

    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            values = img[i - 1:i + 2, j - 1:j + 2]
            values = np.sort(values, axis=None)
            output_img[i, j] = values[4]

    return output_img.astype(np.uint8)


def create_imgs(img, name):
    gx_pw, gx_sb, gx_sc = vertical_components(img)
    gy_pw, gy_sb, gy_sc = horizontal_components(img)

    img_magnitude_pw = gradient_magnitude(gx_pw, gy_pw)
    img_magnitude_sb = gradient_magnitude(gx_sb, gy_sb)
    img_magnitude_sc = gradient_magnitude(gx_sc, gy_sc)

    img_direction_pw = gradient_direction(gx_pw, gy_pw)
    img_direction_sb = gradient_direction(gx_sb, gy_sb)
    img_direction_sc = gradient_direction(gx_sc, gy_sc)

    img_local_max_pw = local_maximum(gx_pw, gy_pw)
    img_local_max_sb = local_maximum(gx_sb, gy_sb)
    img_local_max_sc = local_maximum(gx_sc, gy_sc)

    cv2.imwrite('imagens/saida/Prewitt/' + name + '_Gx.png', gx_pw)
    cv2.imwrite('imagens/saida/Prewitt/' + name + '_Gy.png', gy_pw)
    cv2.imwrite('imagens/saida/Prewitt/' + name + '_magnitude.png', img_magnitude_pw)
    cv2.imwrite('imagens/saida/Prewitt/' + name + '_direction.png', img_direction_pw)
    cv2.imwrite('imagens/saida/Prewitt/' + name + '_local_max.png', img_local_max_pw)

    cv2.imwrite('imagens/saida/Sobel/' + name + '_Gx.png', gx_sb)
    cv2.imwrite('imagens/saida/Sobel/' + name + '_Gy.png', gy_sb)
    cv2.imwrite('imagens/saida/Sobel/' + name + '_magnitude.png', img_magnitude_sb)
    cv2.imwrite('imagens/saida/Sobel/' + name + '_direction.png', img_direction_sb)
    cv2.imwrite('imagens/saida/Sobel/' + name + '_local_max.png', img_local_max_sb)

    cv2.imwrite('imagens/saida/Scharr/' + name + '_Gx.png', gx_sc)
    cv2.imwrite('imagens/saida/Scharr/' + name + '_Gy.png', gy_sc)
    cv2.imwrite('imagens/saida/Scharr/' + name + '_magnitude.png', img_magnitude_sc)
    cv2.imwrite('imagens/saida/Scharr/' + name + '_direction.png', img_direction_sc)
    cv2.imwrite('imagens/saida/Scharr/' + name + '_local_max.png', img_local_max_sc)


if __name__ == '__main__':
    img_a = cv2.imread('imagens/entrada/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)
    img_a = median_filter(median_filter(img_a))
    img_b = cv2.imread('imagens/entrada/chessboard_inv.png', cv2.IMREAD_GRAYSCALE)
    create_imgs(img_a, 'lua')
    create_imgs(img_b, 'chessboard')

import cv2
import math
import numpy as np

Pwx = np.array([ [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Pwy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

Sbx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sby = np.array([[-1, -2, -1], [0, 0, 0], [1, 2 ,1] ])

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
    output_img = np.clip(output_img, 0, 255)

    return output_img.astype(np.uint8)


def gradient_magnitude(img, Gx, Gy):
    Gx = Gx.astype(np.float32)
    Gy = Gy.astype(np.float32)
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    magnitude = np.clip(magnitude, 0, 255)
    magnitude = magnitude.astype(np.uint8)

    return magnitude


def gradient_direction(img, Gx, Gy):
    Gx = Gx.astype(np.float32)
    Gy = Gy.astype(np.float32)
    eps = 10 ** (-8) 
    direction = np.degrees(np.arctan2(Gy, Gx + eps))
    direction = np.clip(direction, 0, 255)    
    direction = direction.astype(np.uint8)

    return direction

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

def create_imgs(img, name):
    Gx_pw, Gx_sb, Gx_sc = vertical_components(img)
    Gy_pw, Gy_sb, Gy_sc = horizontal_components(img)

    img_magnitude_pw = gradient_magnitude(img, Gx_pw, Gy_pw)
    img_magnitude_sb = gradient_magnitude(img, Gx_sb, Gy_sb)
    img_magnitude_sc = gradient_magnitude(img, Gx_sc, Gy_sc)

    img_direction_pw = gradient_direction(img, Gx_pw, Gy_pw)
    img_direction_sb = gradient_direction(img, Gx_sb, Gy_sb)
    img_direction_sc = gradient_direction(img, Gx_sc, Gy_sc)

    cv2.imwrite('imagens/saida/Prewitt/'+name+'_Gx.png', Gx_pw)
    cv2.imwrite('imagens/saida/Prewitt/'+name+'_Gy.png', Gy_pw)
    cv2.imwrite('imagens/saida/Prewitt/'+name+'_magnitude.png', img_magnitude_pw)
    cv2.imwrite('imagens/saida/Prewitt/'+name+'_direction.png', img_direction_pw)

    cv2.imwrite('imagens/saida/Sobel/'+name+'_Gx.png', Gx_sb)
    cv2.imwrite('imagens/saida/Sobel/'+name+'_Gy.png', Gy_sb)
    cv2.imwrite('imagens/saida/Sobel/'+name+'_magnitude.png', img_magnitude_sb)
    cv2.imwrite('imagens/saida/Sobel/'+name+'_direction.png', img_direction_sb)

    cv2.imwrite('imagens/saida/Scharr/'+name+'_Gx.png', Gx_sc)
    cv2.imwrite('imagens/saida/Scharr/'+name+'_Gy.png', Gy_sc)
    cv2.imwrite('imagens/saida/Scharr/'+name+'_magnitude.png', img_magnitude_sc)
    cv2.imwrite('imagens/saida/Scharr/'+name+'_direction.png', img_direction_sc)

if __name__ == '__main__':
    img_a = cv2.imread('imagens/saida/1_a_3.png', cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread('imagens/entrada/chessboard_inv.png', cv2.IMREAD_GRAYSCALE)
    create_imgs(img_a, 'lua')
    create_imgs(img_b, 'chessboard')
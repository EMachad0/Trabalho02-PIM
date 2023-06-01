import cv2
import numpy as np


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


def question_i():
    img = cv2.imread('imagens/entrada/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    a = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])

    b = np.array([[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]])

    c = -1

    laplacian_image_a = apply_kernel(img, a)
    laplacian_image_b = apply_kernel(img, b)

    img_a = img + c * laplacian_image_a
    img_b = img + c * laplacian_image_b

    cv2.imwrite('imagens/saida/2_a.png', img_a)
    cv2.imwrite('imagens/saida/2_b.png', img_b)


if __name__ == '__main__':
    question_i()

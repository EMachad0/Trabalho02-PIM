import cv2
import numpy as np


def apply_kernel(img, kernel):
    sz = kernel.shape[0] // 2

    # add mirror board
    img = cv2.copyMakeBorder(img, sz, sz, sz, sz, cv2.BORDER_REFLECT)

    result = np.zeros_like(img)

    for i in range(sz, img.shape[0] - sz):
        for j in range(sz, img.shape[1] - sz):
            result[i, j] = np.sum(img[i - sz:i + sz + 1, j - sz:j + sz + 1] * kernel)

    # Remove mirror board
    result = result[sz:-sz, sz:-sz]

    return result


def question_i():
    img = cv2.imread('imagens/entrada/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    mean_kernel_3 = np.full((3, 3), 1 / 9)
    mean_kernel_5 = np.full((5, 5), 1 / 25)

    img_3 = apply_kernel(img, mean_kernel_3)
    img_5 = apply_kernel(img, mean_kernel_5)

    cv2.imwrite('imagens/saida/1_a_3.png', img_3)
    cv2.imwrite('imagens/saida/1_a_5.png', img_5)


def question_ii():
    img = cv2.imread('imagens/entrada/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    def create_gaussian_kernel(size, sigma):
        sz = size // 2
        gaussian_kernel = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                x = i - sz
                y = j - sz
                gaussian_kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        return gaussian_kernel

    gaussian_kernel_06 = create_gaussian_kernel(3, 0.6)
    gaussian_kernel_10_a = create_gaussian_kernel(3, 1.0)
    gaussian_kernel_10_b = create_gaussian_kernel(7, 1.0)

    img_06 = apply_kernel(img, gaussian_kernel_06)
    img_10_a = apply_kernel(img, gaussian_kernel_10_a)
    img_10_b = apply_kernel(img, gaussian_kernel_10_b)

    cv2.imwrite('imagens/saida/1_b_06.png', img_06)
    cv2.imwrite('imagens/saida/1_b_10_a.png', img_10_a)
    cv2.imwrite('imagens/saida/1_b_10_b.png', img_10_b)


if __name__ == '__main__':
    question_i()
    question_ii()

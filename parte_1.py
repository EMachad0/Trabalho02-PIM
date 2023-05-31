import cv2
import numpy as np


def apply_kernel(img, kernel):
    sz = kernel.shape[0] // 2

    # add mirror board
    img = cv2.copyMakeBorder(img, sz, sz, sz, sz, cv2.BORDER_REFLECT)

    for i in range(sz, img.shape[0] - sz):
        for j in range(sz, img.shape[1] - sz):
            img[i, j] = np.sum(img[i - sz:i + sz + 1, j - sz:j + sz + 1] * kernel)

    # Remove mirror board
    img = img[sz:-sz, sz:-sz]

    return img


def question_i():
    img = cv2.imread('imagens/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    mean_kernel_3 = np.full((3, 3), 1 / 9)
    mean_kernel_5 = np.full((5, 5), 1 / 9)

    img_3 = apply_kernel(img, mean_kernel_3)
    img_5 = apply_kernel(img, mean_kernel_5)

    cv2.imwrite('imagens/1_a_3.png', img_3)
    cv2.imwrite('imagens/1_a_5.png', img_5)


def question_ii():
    img = cv2.imread('imagens/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    def create_gaussian_kernel(size, sigma):
        gaussian_kernel = np.zeros((size, size))
        for i in range(3):
            for j in range(3):
                x = i - 1
                y = j - 1
                gaussian_kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return gaussian_kernel

    gaussian_kernel_06 = create_gaussian_kernel(3, 0.6)
    gaussian_kernel_10 = create_gaussian_kernel(3, 1.0)

    img_06 = apply_kernel(img, gaussian_kernel_06)
    img_10 = apply_kernel(img, gaussian_kernel_10)

    cv2.imwrite('imagens/1_b_06.png', img_06)
    cv2.imwrite('imagens/1_b_10.png', img_10)


def question_iii():
    pass


if __name__ == '__main__':
    question_i()
    question_ii()
    question_iii()

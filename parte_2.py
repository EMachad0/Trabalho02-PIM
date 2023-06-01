import cv2
import numpy as np


def question_i():
    img = cv2.imread('imagens/entrada/Lua1_gray.jpg', cv2.IMREAD_GRAYSCALE)

    a = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])

    b = np.array([[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]])

    c = -1

    laplacian_image_a = cv2.filter2D(src=img, ddepth=-1, kernel=a)
    laplacian_image_b = cv2.filter2D(src=img, ddepth=-1, kernel=a)

    img_a = img + c * laplacian_image_a
    img_b = img + c * laplacian_image_b

    cv2.imwrite('imagens/saida/2_a.png', img_a)
    cv2.imwrite('imagens/saida/2_b.png', img_b)


if __name__ == '__main__':
    question_i()

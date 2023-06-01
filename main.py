import cv2
import numpy as np
import math
import sys
from skimage.filters import laplace

#Alunos: Tiago Gonçalves da Silva e Luis Felipe Moro Coelho

def Init ():
    return

def main():
    img = cv2.imread("Original.bmp", cv2.IMREAD_COLOR)
    mask = cv2.imread("Mascara2.bmp", cv2.IMREAD_GRAYSCALE)

    row, col, cha = img.shape
    mask = cv2.resize(mask.astype(np.uint8),(col, row), cv2.INTER_NEAREST)

    working_mask = np.copy(mask)
    working_image = np.copy(img)

    i_mask = 1 - working_mask
    rgb_i_mask = working_mask.reshape(row, col, 1).repeat(3, axis=2)
    cv2.imwrite('resultado_rgbmascara.bmp', rgb_i_mask* 255)
    working_image = working_image * rgb_i_mask

    cv2.imwrite('resultado_mascara.bmp', working_image* 255)

    front = (laplace(working_mask) > 0).astype('uint8')

    cv2.imwrite('resultado_laplace.bmp', front* 255)

    #white_region = 1 - (working_mask * front) * 255
    #rgb_white_region = white_region.reshape(row, col, 1).repeat(3, axis=2)

    #cv2.imwrite('resultado_whiteregion.bmp', rgb_white_region* 255)
    #working_image *= rgb_white_region

    #cv2.imwrite('resultado_parcial1.bmp', working_image* 255)

    border_left = False
    #while border_left == True:

        # Identifica as bordas da máscara ( Pode ser feito com o Laplace)
        # Calcula/Atualiza as prioridades 
        # Procura o pixel com maior prioridade 
        # Atualiza o pixel
if __name__ == '__main__':
    main()

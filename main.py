import cv2
import numpy as np
import math
import sys
from skimage.filters import laplace

#Alunos: Tiago Gonçalves da Silva e Luis Felipe Moro Coelho

def Init ():
    return

def main():
    img = cv2.imread("elephant.bmp", cv2.IMREAD_COLOR)
    mask = cv2.imread("elephant-mask.bmp", cv2.IMREAD_GRAYSCALE)

    row, col, cha = img.shape

    working_mask = np.copy(mask)
    working_image = np.copy(img)

    rgb_mask = cv2.merge((working_mask, working_mask, working_mask))
    working_image = working_image * rgb_mask
    cv2.imwrite('1-resultado_mascara.bmp', working_image* 255)
    #Para testar como esta ocorrendo primeiramente, comente o For e execute somente o que está dentro dele
    for i in range(1):  #Será substituido pelo while que rodara até não termos pixels brancos dentro da matriz front

        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.dilate(working_mask, kernel)
        cv2.imwrite('2-resultado_erosão.bmp', eroded_mask)

        front = (1 - working_mask) - (1 - eroded_mask)
        cv2.imwrite('3-pixels_investigados.bmp', front)

        #Usaria a lista front para calcular as prioridades

        #Após calcular as prioridades partimos para o filtro na imagem
        
        #Utilizamos a working image para utilizar das vizinhanças, pois ela primeiramente ja estará com o buraco na imagem

        #Então atualizamos a working image e a máscara como abaixo
        working_mask = eroded_mask
        cv2.imwrite('4-nova_mascara.bmp', working_mask)

    rgb_mask = cv2.merge((working_mask, working_mask, working_mask))
    mock_resultado_final =  img = img * rgb_mask
    cv2.imwrite('5-mock_final.bmp', mock_resultado_final* 255)

    #while border_left == True:

        # Identifica as bordas da máscara ( Pode ser feito com o Laplace)
        # Calcula/Atualiza as prioridades 
        # Procura o pixel com maior prioridade 
        # Atualiza o pixel
if __name__ == '__main__':
    main()

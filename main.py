import cv2
import numpy as np
import math
import sys
#from skimage.filters import laplace

#Alunos: Tiago Gonçalves da Silva e Luis Felipe Moro Coelho
PSI_SIZE = 9

def Init ():
    return

def calcConfidence(row, col, border, confidence):

    auxConfidence = np.zeros(confidence.shape, np.float32)

    lim = PSI_SIZE//2

    # Pass the entire image
    for i in range(row):
        for j in range(col):
            # If the analysed pixel is border
            if border[i][j] == 1:
                area = 0
                for x in range(-lim, lim+1):
                    for y in range(-lim, lim+1):
                        # If the pixel is inside the image
                        if i+x >= 0 and i+x <= row and j+y >= 0 and j+y <= col:
                            # Sums the confidence values in an auxiliar matrix
                            auxConfidence[i][j] += confidence[i+x][j+y]
                            area += 1
                auxConfidence[i][j] /= area
    confidencePrint = np.copy(auxConfidence)
    cv2.normalize(auxConfidence, confidencePrint, 0, 1, cv2.NORM_MINMAX)
    
    # Print only the max confidence pixels
    #confidencePrint = np.where(confidencePrint == 1, 1, 0)

    # Print the resulting matix
    cv2.imwrite('4-confidence.bmp', confidencePrint*255)
    return auxConfidence

def main():
    img = cv2.imread("elephant.bmp", cv2.IMREAD_COLOR).astype(np.float32) / 255
    mask = cv2.imread("elephant-mask.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    row, col, cha = img.shape

    working_mask = np.copy(mask)
    confidence = np.copy(mask)
    working_image = np.copy(img)

    rgb_mask = cv2.merge((working_mask, working_mask, working_mask))
    working_image = working_image * rgb_mask
    cv2.imwrite('1-resultado_mascara.bmp', working_image* 255)
    #Para testar como esta ocorrendo primeiramente, comente o For e execute somente o que está dentro dele
    for i in range(1):  #Será substituido pelo while que rodara até não termos pixels brancos dentro da matriz front

        kernel = np.ones((3, 3), np.float32)
        eroded_mask = cv2.dilate(working_mask, kernel)
        cv2.imwrite('2-resultado_erosao.bmp', eroded_mask*255)

        border = (1 - working_mask) - (1 - eroded_mask)
        cv2.imwrite('3-pixels_investigados.bmp', border*255)

        #Usaria a lista front para calcular as prioridades
        auxConfidence = calcConfidence(row, col, border, confidence)

        #Após calcular as prioridades partimos para o filtro na imagem
        
        #Utilizamos a working image para utilizar das vizinhanças, pois ela primeiramente ja estará com o buraco na imagem

        #Então atualizamos a working image e a máscara como abaixo
        working_mask = eroded_mask
        #cv2.imwrite('nova_mascara.bmp', working_mask*255)

    rgb_mask = cv2.merge((working_mask, working_mask, working_mask))
    mock_resultado_final =  img = img * rgb_mask
    #cv2.imwrite('mock_final.bmp', mock_resultado_final* 255)

    #while border_left == True:

        # Identifica as bordas da máscara ( Pode ser feito com o Laplace)
        # Calcula/Atualiza as prioridades 
        # Procura o pixel com maior prioridade 
        # Atualiza o pixel
if __name__ == '__main__':
    main()

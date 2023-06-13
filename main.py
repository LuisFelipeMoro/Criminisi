import cv2
import numpy as np

#Alunos: Tiago Gonçalves da Silva e Luis Felipe Moro Coelho

# The size of the patch
PSI_SIZE = 9

# If you want to print the images
PRINT = True

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

    if PRINT:
        confidencePrint = np.copy(auxConfidence)
        cv2.normalize(auxConfidence, confidencePrint, 0, 1, cv2.NORM_MINMAX)
        # Print the resulting matix
        cv2.imwrite('4.1-confidence.bmp', confidencePrint*255)
        
        # Print only the max confidence pixels
        confidencePrint = np.where(confidencePrint == 1, 1, 0)
        cv2.imwrite('4.2-confidenceMax.bmp', confidencePrint*255)

    return auxConfidence

def calcData(row, col, border, working_image, mask):

    imgGray = np.array(cv2.cvtColor(working_image,
                       cv2.COLOR_BGR2GRAY), dtype=np.float64)

    data = np.zeros(border.shape, np.float32)

    kernel_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])
    kernel_y = np.array([[-3, -10, -3],
                         [0, 0, 0],
                         [3, 10, 3]])
    alpha = 1

    # Pass the entire image
    for i in range(row):
        for j in range(col):
            # If the analysed pixel is border
            if border[i][j] == 1:
                gray_patch = imgGray[i-1:i+2, j-1:j+2]
                isophote = np.nan_to_num(np.array(np.gradient(gray_patch))) 
                isophote = np.array([np.max(isophote[0]),np.max(isophote[1])])
                normal_x = np.sum(np.multiply(mask[i-1:i+2, j-1:j+2], kernel_x))
                normal_y = np.sum(np.multiply(mask[i-1:i+2, j-1:j+2], kernel_y))
                normal = np.array([normal_x, normal_y])
                normal = normal/np.linalg.norm(normal)
                data[i][j] = abs(np.dot(isophote, normal))/alpha + 0.001

    if PRINT:
        dataPrint = np.copy(data)
        cv2.normalize(data, dataPrint, 0, 1, cv2.NORM_MINMAX)
        # Print the resulting matix
        cv2.imwrite('5.1-data.bmp', dataPrint*255)

        # Print only the max confidence pixels
        dataPrint = np.where(dataPrint == 1, 1, 0)
        cv2.imwrite('5.2-dataMax.bmp', dataPrint*255)

    return data

def calcPriority(auxConfidence, data):
    priority = auxConfidence * data

    maxValue = -1
    px = -1
    py = -1
    for i in range(priority.shape[0]):
        for j in range(priority.shape[1]):
            if priority[i][j] > maxValue:
                px = i
                py = j
                maxValue = priority[i][j]

    if PRINT:
        priorityPrint = np.copy(priority)
        cv2.normalize(priority, priorityPrint, 0, 1, cv2.NORM_MINMAX)
        # Print the priority matix
        cv2.imwrite('6.1-priority.bmp', priorityPrint*255)

        # Print only the max confidence pixels
        priorityPrint = np.where(priorityPrint == 1, 1, 0)
        cv2.imwrite('6.2-priorityMax.bmp', priorityPrint*255)

    return px,py

def main():
    img = cv2.imread("elephant.bmp", cv2.IMREAD_COLOR).astype(np.float32) / 255
    mask = cv2.imread("elephant-mask.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    row, col, cha = img.shape

    working_mask = np.copy(mask)
    confidence = np.copy(mask)
    working_image = np.copy(img)

    rgb_mask = cv2.merge((working_mask, working_mask, working_mask))
    working_image = working_image * rgb_mask
    if PRINT:
        cv2.imwrite('1-resultado_mascara.bmp', working_image* 255)

    #Para testar como esta ocorrendo primeiramente, comente o For e execute 
    # somente o que está dentro dele

    #Será substituido pelo while que rodara até não termos pixels brancos dentro da matriz front
    for i in range(1):  #Será substituido pelo while que rodara até não termos pixels brancos dentro da matriz front

        kernel = np.ones((3, 3), np.float32)
        eroded_mask = cv2.dilate(working_mask, kernel)
        if PRINT:
            cv2.imwrite('2-resultado_erosao.bmp', eroded_mask*255)

        # Is the matrix that contain the borders of the mask
        border = (1 - working_mask) - (1 - eroded_mask)
        if PRINT:
            cv2.imwrite('3-pixels_investigados.bmp', border*255)

        # Calculates the confidence of all the pixels and put into a matrix
        auxConfidence = calcConfidence(row, col, border, confidence)
        
        # Calculates the data of all the pixels and put into a matrix
        data = calcData(row, col, border, working_image, working_mask)
        
        # Calculates the priority of all the pixels
        # returns the coordinates of the best pixel
        px,py = calcPriority(auxConfidence, data)
        
        #Utilizamos a working image para utilizar das vizinhanças, 
        #pois ela primeiramente ja estará com o buraco na imagem
        #Encontramos a janela da imagem com melhor match para o nosso patch
        #Utilizamos soma das diferenças quadradas, para imagens coloridas 
        #trataremos das cores no espaço lab



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

import cv2
import numpy as np
import time 

#Alunos: Tiago Gonçalves da Silva e Luis Felipe Moro Coelho

# The size of the patch
PSI_SIZE = 9
lim = PSI_SIZE//2

# If you want to print the images
PRINT = True

def Init ():
    return

def calcConfidence(border, confidence):

    auxConfidence = np.zeros(confidence.shape, np.float32)

    lim = PSI_SIZE//2

    # Pass the entire image
    for i in range(border.shape[0]):
        for j in range(border.shape[1]):
            # If the analysed pixel is border
            if border[i][j] == 1:
                area = 0
                for x in range(-lim, lim+1):
                    for y in range(-lim, lim+1):
                        # If the pixel is inside the image
                        if (i+x >= 0 and i+x <= border.shape[0] and 
                            j+y >= 0 and j+y <= border.shape[1]):
                            # Sums the confidence values in an auxiliar matrix
                            auxConfidence[i][j] += confidence[i+x][j+y]
                            area += 1
                auxConfidence[i][j] /= area

    return auxConfidence

def calcData(border, working_image, mask):

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
    for i in range(border.shape[0]):
        for j in range(border.shape[1]):
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

    return data

def calcPriority(auxConfidence, data):
    priority = auxConfidence * data

    max_value = -1
    best_coords = None

    for i, row in enumerate(priority):
        for j, value in enumerate(row):
            if value > max_value:
                max_value = value
                best_coords = (i, j)

    print(f"Best priority coordinates: {best_coords}")
    return best_coords

def biggestDif(p1, p2):
    db = abs(p1[0] - p2[0])
    dg = abs(p1[1] - p2[1])
    dr = abs(p1[2] - p2[2])

    return max(db, dg, dr)

def calcDif(i, j, px, py, mask, img):
    dif = 0

    for x in range(-lim, lim+1):
        for y in range(-lim, lim+1):
            if mask[i+x][j+y] == 0 :
                return False, -1
            if mask[px+x][py+y] == 1:
                aux = biggestDif(img[i+x][j+y], img[px+x][py+y])
                dif += pow(aux, 2)

    return True, np.sqrt(dif)

def calcQRGB(px, py, mask, img):

    qx = -1
    qy = -1
    minDif = float("inf")

    # Pass the entire image, ignoring borders
    for i in range(lim, mask.shape[0]-lim, PSI_SIZE):
        for j in range(lim, mask.shape[1]-lim):
                # Calculates the difference between the window and the patchP
                windowNotInMask, dif = calcDif(i, j, px, py, mask, img)
                if windowNotInMask == True and dif < minDif:
                    minDif = dif
                    qx = i
                    qy = j
        
                
    return qx, qy

def calcMaskArea(mask):
    return np.count_nonzero(mask == 0)

def main():

    start_t = time.perf_counter()
    img = cv2.imread("elephant.bmp", cv2.IMREAD_COLOR).astype(np.float32) / 255
    mask = cv2.imread("elephant-mask.bmp", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    working_mask = np.copy(mask)
    maskArea = calcMaskArea(working_mask)
    confidence = np.copy(mask)
    working_image = np.copy(img)
    kernel = np.ones((3, 3), np.float32)

    rgb_mask = cv2.merge((working_mask, working_mask, working_mask))
    working_image = working_image * rgb_mask

    eroded_mask = cv2.dilate(working_mask, kernel)

    # The matrix that contain the borders of the mask
    border = (1 - working_mask) - (1 - eroded_mask)
    cv2.imwrite('1-[Border]initial_border.bmp', border*255)

    #while any(1 in row for row in border): 
    while maskArea > 0:

        img_lab = cv2.cvtColor(working_image, cv2.COLOR_BGR2LAB)
        # Calculates the confidence of all the pixels and put into a matrix
        auxConfidence = calcConfidence(border, confidence)
        cv2.imwrite('2-[Border Analysis]-resultado_border.bmp', border*255)

        # Calculates the data of all the pixels and put into a matrix
        data = calcData(border, working_image, working_mask)
        cv2.imwrite('3-[Border Analysis]-resultado_data.bmp', data* 255)
        
        # returns the coordinates of the best pixel
        px,py = calcPriority(auxConfidence, data)
        patchPrint = np.copy(working_image)
        cv2.rectangle(patchPrint, (py-lim, px-lim), (py+lim, px+lim), (0, 0, 255))
        cv2.imwrite('4-[Border Analysis]-patchP escolhido.bmp', patchPrint*255)

        start_t = time.time()
        qx,qy = calcQRGB(px, py, working_mask, working_image)
        print(time.time() - start_t)
        print(f"Best patch RGB coordinates: {qx, qy}")
        
        patchPrint = np.copy(working_image)
        cv2.rectangle(patchPrint, (qy-lim, qx-lim), (qy+lim, qx+lim), (0, 0, 255))
        cv2.imwrite('5-[Border Analysis]-patchQ escolhido.bmp', patchPrint*255)

        for i in range(-lim, lim + 1): 
            for j in range(-lim, lim + 1):
                if working_mask[px + i][py + j] == 0:
                    working_image[px + i][py + j] = working_image[qx + i][qy + j]
                    working_mask[px + i][py + j] = 1
                    border[px + i][py + j] = 0
                    confidence[px + i][py + j] = auxConfidence[px + i][py + j]
                    maskArea -= 1

        cv2.imwrite('6-[Patching]-resultado_patching.bmp', working_image* 255)
        cv2.imwrite('7-[Patching]-mask_without_patch.bmp', working_mask* 255)

        eroded_mask = cv2.dilate(working_mask, kernel)
        border = (1 - working_mask) - (1 - eroded_mask)


    #cv2.imwrite(f'{k}iteration_mask.bmp', working_mask*255)

    end_t = time.perf_counter()
    print(f"{start_t - end_t} seconds")

if __name__ == '__main__':
    main()

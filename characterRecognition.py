#%%

import numpy as np
import os  
import numpy as np
import cv2  as cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt  



def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def operate_through_pictures(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (JPEG or PNG)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # Check if image is loaded successfully
            if image is None:
                print(f"Error: Unable to load image from {image_path}")
                continue
            prepare_image(image,filename, output_folder)
            # Define the region of interest (ROI)

def FilterContours(contours_Tablica):
    inner_contours = []
    outter_contours = []
    together_contours = []
    for i in range (len(contours_Tablica)):
        outter_contours.append(cv2.boundingRect(contours_Tablica[i]))
        ox,oy,ow,oh = cv2.boundingRect(contours_Tablica[i])
        for j in range (len(contours_Tablica)):
            if (i != j):
                ix,iy,iw,ih = cv2.boundingRect(contours_Tablica[j])
                if (ox<ix) & ((ox + ow) > (ix + iw)) & (oy<iy) & ((oy+oh)>(iy+ih)):
                    #print("DODAO",contours_Tablica[j]);
                    inner_contours.append(cv2.boundingRect(contours_Tablica[j]))
                #Objedinjavanje preklapajucih kontura
                if (((ox - 7) < ix < (ox + 7)) & ((ow - 7) < iw < (ow + 7))):
                    vstacked_array = np.vstack((contours_Tablica[i], contours_Tablica[j]))
                    together_contours.append(cv2.boundingRect(vstacked_array))
                    inner_contours.append(cv2.boundingRect(contours_Tablica[i]))
                    inner_contours.append(cv2.boundingRect(contours_Tablica[j]))
        
    result = [item for item in outter_contours if item not in inner_contours]
    for contour in together_contours:
        result.append(contour)
    
    return result.sort(key=lambda x: x[0])
    
def drawContoursAndExport(contours, image, filename):
    for contour in contours:
        x, y, w, h = contour
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)   
            
        base_filename, ext = os.path.splitext(filename)
        outname = f"{base_filename}{ext}"
        output_path = os.path.join(output_folder, outname)
        cv2.imwrite(output_path, image)

        print(f"Cropped image saved to {output_path}")
            

def findPlatesContours(contours):
    contours_Tablica = []
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size  # Obrnuo sam parametre , Prvo ide visina ??!!?!?!
        xx, yy, w, h = cv2.boundingRect(contour)
        x, y = center
        if w > 10 and w<100 and h > 40 and h<130:  # uslov da kontura pripada bar-kodu
            #if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
                contours_Tablica.append(contour)  # ova kontura pripada bar-kodu
    return contours_Tablica

def prepare_image(img, filename, output_folder):
    min_number_of_contours = 7
    kernel = np.ones((2, 2), np.uint8)
    img_barcode_gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    #plt.imshow(img_barcode_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_barcode_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, 15)
    dilated_image = cv2.erode(image_barcode_bin, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_c = img.copy()
   
    contours_Tablica = findPlatesContours(contours) # ovde ce biti samo konture koje pripadaju bar-kodu
    
    if (len(contours_Tablica)<min_number_of_contours):
        dilated_image = dilated_image = cv2.erode(image_barcode_bin, kernel, iterations=2)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_Tablica = findPlatesContours(contours)

        if (len(contours_Tablica)<min_number_of_contours):
            dilated_image = dilated_image = cv2.erode(image_barcode_bin, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours_Tablica = findPlatesContours(contours)
        else:
            filtered_contours = FilterContours(contours_Tablica)
            drawContoursAndExport(filtered_contours,img_c,filename) 
            
    else:
        filtered_contours = FilterContours(contours_Tablica) 
        drawContoursAndExport(filtered_contours,img_c,filename) 
     
# Example usage
input_folder = '../Plates/'
output_folder = '../PlatesMarked/'

operate_through_pictures(input_folder, output_folder)        

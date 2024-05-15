#%%

import numpy as np
import os  
import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt  

#%%
# prikaz vecih slika 


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def crop_images_in_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (JPEG or PNG)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = load_image(image_path)

            # Check if image is loaded successfully
            if image is None:
                print(f"Error: Unable to load image from {image_path}")
                continue
            prepare_image(image,filename, output_folder)
            # Define the region of interest (ROI)
            


def prepare_image(img, filename, output_folder):
    #display_image(img)
    #img_gs = image_gray(img)
    #_, black_mask = cv2.threshold(img_gs, 50, 255, cv2.THRESH_BINARY_INV)
    #_, white_mask = cv2.threshold(img_gs, 200, 255, cv2.THRESH_BINARY)
    #display_image(black_mask)
    #display_image(white_mask)

    original = img
    img_barcode_gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # konvert u grayscale
    #plt.imshow(img_barcode_gs, 'gray')
    image_barcode_bin = cv2.adaptiveThreshold(img_barcode_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, 15)
    #display_image(image_barcode_bin)
    contours, hierarchy = cv2.findContours(image_barcode_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_c = img.copy()
    cv2.drawContours(img_c, contours, -1, (255, 0, 0), 1)
    #display_image(img_c)

    contours_Tablica = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
            center, size, angle = cv2.minAreaRect(
                contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
            width, height = size  # Obrnuo sam parametre , Prvo ide visina ??!!?!?!
            xx, yy, w, h = cv2.boundingRect(contour)
            x, y = center
            if w > 350 and w<970 and h > 100 and h<300:  # uslov da kontura pripada bar-kodu
                #if (xx > 200 and xx < 400 and yy > 210 and yy < 380):
                    contours_Tablica.append(contour)  # ova kontura pripada bar-kodu

    num_elements = len(contours_Tablica)

    # Iterate through the list
    for i, contour in enumerate(contours_Tablica):
        print(f"Processing contour {i + 1}/{num_elements}")

        # Get the bounding rectangle for the current contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the image using the bounding rectangle
        roi = original[y:y+h, x:x+w]

        # Save the cropped image with a new filename
        base_filename, ext = os.path.splitext(filename)
        outname = f"{base_filename}_{i+1}{ext}"
        output_path = os.path.join(output_folder, outname)
        cv2.imwrite(output_path, roi)

        print(f"Cropped image saved to {output_path}")
            #cv2.rectangle(img_c, (x, y), (x+w, y+h), (0, 255, 255), 3)
            #cv2.drawContours(img_c, contours_Tablica[0], -1, (0, 255, 0), 3)
            #display_image(img_c)

# Example usage
input_folder = 'input_folder_path'
output_folder = 'output_folder_path'
x, y = 204, 5  # Top-left corner coordinates
width, height = 23, 40  # Width and height of the cropping rectangle

crop_images_in_folder('../Images/', '../Plates/')        

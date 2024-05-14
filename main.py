# %%
import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt  

 
# prikaz vecih slika 
matplotlib.rcParams['figure.figsize'] = 16,12

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



# %%
path = '../Images/NS145NG.jpg' # ucitavanje slike sa diska
img = load_image(path)
display_image(img)

# %%
#img_crop = img[240:, :]  
#display_image(img_crop)

# %%
img_gs = image_gray(img)
display_image(img_gs)

# %%
mean_value = cv2.mean(img_gs)
img_mean = img_gs > mean_value[0]
display_image(img_mean)

#%%
edges = cv2.Canny(img_gs, 50, 50)
display_image(edges)


# %%
img_new = np.uint8(img_mean)
sure_bg = cv2.erode(img_new, kernel, iterations=1)
img_in = invert(sure_bg)
#display_image(sure_bg)

# %%
img_open = cv2.dilate(sure_bg, kernel, iterations=2)
#display_image(img_open)

# %%
kernel = np.ones((3,3), np.uint8) # strukturni element 3x3 blok
opening = cv2.morphologyEx(img_open, cv2.MORPH_OPEN, kernel, iterations = 1) # otvaranje
#display_image(opening)

# %%




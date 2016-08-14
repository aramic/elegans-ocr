# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import cv2
#np.set_printoptions(threshold=999999)

abeceda = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'DŽ', 'Đ', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'LJ', 'M', 'N', 'NJ', 'O', 'P', 'R', 'S', 'Š', 'T', 'U', 'V', 'Z', 'Ž']

# Set up the training image. 
# It is an alphabetical tablet of ladybugs.
img1 = cv2.imread('calibration/tablet.jpg')
bubamara = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
 
# Set up the user's input image to check against.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
img2 = cv2.imread(args["image"])
#img2 = cv2.imread('img/banatskidvor.jpg')
input_name = os.path.basename(args["image"])
#input_name = os.path.basename('img/banatskidvor.jpg')
input_img = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# Make the input image the same size as the training image.
input_img = cv2.resize(input_img, (bubamara.shape[0],bubamara.shape[1]), interpolation = cv2.INTER_AREA)

# Now we split the images to 30 cells, each 100x100 in size. (6x5 grid)
# 30 corresponds to each letter of the Bosnian-Serbian-Croatian alphabet.
bubamara_cells = [np.hsplit(row,100) for row in np.vsplit(bubamara,100)]
input_img_cells = [np.hsplit(row,100) for row in np.vsplit(input_img,100)]

# Make both into Numpy arrays. Their sizes will be (100,100,6,5)
# Total area: 300000
x1 = np.array(bubamara_cells)
x2 = np.array(input_img_cells)

# Now we prepare train_data and test_data.
train = x1[:,:100].reshape(-1,100).astype(np.float32) # Size = (3000,100)
test = x2[:,:100].reshape(-1,100).astype(np.float32) # Size = (3000,100)

# Create labels for the first 30 cells.
k = np.arange(30)
train_labels = np.repeat(k,100)[:,np.newaxis] # Length of labels will be 30

# Initiate kNN and train the data.
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=2)
result = result[:,:30].reshape(-1,1).astype(np.int)

# Turn the result into a Unicode string
s = ""
for item in result:
	s += abeceda[item[0]]

# Save text to a file
output_filename = 'output/' + input_name + ".txt"
output_text = open(output_filename, 'w')
output_text.write(s)
output_text.close()
import argparse
import os
import numpy as np
import cv2
#np.set_printoptions(threshold=999999)

# Set up the training image. We are using an image 
# of C Elegans observed under a microscope.
img1 = cv2.imread('img/c-elegans-1600x1600.jpg')
eleg = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
 
# Set up the user's input image to check against.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
img2 = cv2.imread(args["image"])
input_name = os.path.basename(args["image"])
input_img = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


# Make the input image the same size as the training image.
input_img = cv2.resize(input_img, (eleg.shape[0],eleg.shape[1]), interpolation = cv2.INTER_AREA)

# Now we split the images to 256 cells, each 100x100 in size.
# 256 corresponds to each of the code points in Latin-1.
eleg_cells = [np.hsplit(row,100) for row in np.vsplit(eleg,100)]
input_img_cells = [np.hsplit(row,100) for row in np.vsplit(input_img,100)]

# Make both into Numpy arrays. Their sizes will be (100,100,16,16)
x1 = np.array(eleg_cells)
x2 = np.array(input_img_cells)

# Now we prepare train_data and test_data.
train = x1[:,:100].reshape(-1,100).astype(np.float32) # Size = (25600,100)
test = x2[:,:100].reshape(-1,100).astype(np.float32) # Size = (25600,100)

# Create labels for the first 256 cells.
k = np.arange(256)
train_labels = np.repeat(k,100)[:,np.newaxis] # Length of labels will be 25600

# Initiate kNN and train the data.
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=2)
result = result[:,:25600].reshape(-1,1).astype(np.int)

# Turn the result into a Unicode string
s = ""
for item in result:
	s += unichr(item[0])

# Save text to a file
filename = input_name + ".txt"
text = open(filename, 'w')
text.write(s.encode('utf8'))
text.close()
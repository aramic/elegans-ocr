import numpy as np
import cv2
from matplotlib import pyplot as plt
np.set_printoptions(threshold=999999)

img1 = cv2.imread('img/c-elegans-1600x1600.jpg')
eleg = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('img/AS14tracks.jpg')
apollo = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# make test image same size as train image
apollo = cv2.resize(apollo, (eleg.shape[0],eleg.shape[1]), interpolation = cv2.INTER_AREA)

# Now we split the images to 256 cells, each 100x100 size
# 256 corresponds to each of the code points in Latin-1
eleg_cells = [np.hsplit(row,100) for row in np.vsplit(eleg,100)]
apollo_cells = [np.hsplit(row,100) for row in np.vsplit(apollo,100)]

# print eleg_cells

# Make it into a Numpy array. It size will be (100,100,16,16)
x1 = np.array(eleg_cells)
x2 = np.array(apollo_cells)

# # Now we prepare train_data and test_data.
train = x1[:,:100].reshape(-1,100).astype(np.float32) # Size = (25600,100)
test = x1[:,:100].reshape(-1,100).astype(np.float32) # Size = (25600,100)

# Create labels for first 256 cells.
k = np.arange(256)
train_labels = np.repeat(k,100)[:,np.newaxis] # Length of labels will be 25600

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=2)
result = result[:,:25600].reshape(-1,1).astype(np.int)

# Turn the result into a Unicode string
s = ""
for item in result:
	s += unichr(item[0])

# Save text to a file
text = open('elegans-ocr.txt', 'w')
text.write(s.encode('utf8'))
text.close()
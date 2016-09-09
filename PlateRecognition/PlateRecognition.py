import cv2
import numpy as np
import matplotlib.pyplot as plt
import OCR_NN
import OCR_RandomForest as forest

# Load image and convert to gray scale
plate = cv2.imread('numbers_1.jpg')
plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

# Threshold the grayscale image to get binary image
ret, binary = cv2.threshold(plate_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours to seperate each letter
contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Clone the input image for output purpose
output = plate.copy()

# Create an array to store bounding rectangles around contours found above
rect = []

# Create an array of images to store each letter seperately
numbers = []

# Kernel used for erosing
kernel = np.ones((3,3), 'uint8')

# Loop over each contours
for i in range(len(contours)):

	tempRect = cv2.boundingRect(contours[i])

	# Check the area of the bounding rectangle to eliminate small contour
	if (tempRect[2] * tempRect[3] >= 100 and tempRect[2] * tempRect[3] < 3600):
		# If a contour has a parent but no child, so it's a hole inside a letter
		if (hierarchy[0][i][3] != -1 & hierarchy[0][i][2] == -1):
			continue
		else:
			rect.append(tempRect)
			# Draw rectangle on the output image
			cv2.rectangle(output, (tempRect[0], tempRect[1]), (tempRect[0] + tempRect[2], tempRect[1] + tempRect[3]), (0, 255, 0), 2)

			# Crop the bounding rectangle from the original binary image
			number = plate_gray[tempRect[1]:tempRect[1]+tempRect[3], tempRect[0]:tempRect[0]+tempRect[2]]
			
			# Here I just wanted to add some margin to the cropped ROI
			dh = int(tempRect[3] * 0.5)
			dw = int(tempRect[2] * 0.7)

			# Check if it's '1' letter for '1' letters have large height/width ratio
			ratio = tempRect[3] / float(tempRect[2])
			
			if (ratio > 2.6):
				dw = tempRect[2] * 4
			new_number = np.ones((tempRect[3] + dh, tempRect[2] + dw), 'uint8') * 255
			new_number[dh/2:tempRect[3]+dh/2, dw/2:tempRect[2]+dw/2] = number
			
			numbers.append(cv2.subtract(255, new_number))

# Create an array to store 784x1 images used for training
test_data = []

# Create an array to store 28x28 images
resized_numbers = []

for i, number in enumerate(numbers):
	number = cv2.resize(number, (28, 28))
	resized_numbers.append(number)
	# cv2.imshow('Roi {}'.format(i), number)
	# The 28x28 must be reshaped to be accepted by the classifier
	test_data.append(np.reshape(number/255., (784,1)))

# np.savetxt('test.csv', test_data)

"""
# Create an instance of RandomForest classifier
clf = forest.initialize()

# Train the model using MNIST dataset
clf = forest.train_evaluate(clf)
"""

clf = OCR_NN.Network([784, 100, 10])
results = clf.predict(test_data)

print(results)

for i, result in enumerate(results):
	# Put the result on each letter of output image
	cv2.putText(output, str(int(result)), (rect[i][0]+rect[i][2], rect[i][1]+rect[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# For visualizing each letter
	#cv2.imshow('Roi {}'.format(i), resized_numbers[i])

cv2.imshow('Input', plate)
cv2.imshow('Result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
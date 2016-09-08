import cv2
import numpy as np
import matplotlib.pyplot as plt
import OCR_NN

plate = cv2.imread('Plate.jpg')
plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(plate_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#binary_inv = cv2.subtract(255, binary)
output = plate.copy()

#cv2.drawContours(plate, contours, -1, (0, 255, 0), 2)
rect = []
numbers = []
j = 0
kernel = np.ones((3,3), 'uint8')

for i in range(len(contours)):
	tempRect = cv2.boundingRect(contours[i])
	if (tempRect[2] * tempRect[3] >= 1000 and tempRect[2] * tempRect[3] < 3600):
		rect.append(tempRect)
		cv2.rectangle(output, (tempRect[0], tempRect[1]), (tempRect[0] + tempRect[2], tempRect[1] + tempRect[3]), (0, 255, 0), 2)
		number = binary[tempRect[1]:tempRect[1]+tempRect[3], tempRect[0]:tempRect[0]+tempRect[2]]
		
		new_number = np.zeros((tempRect[3] + 30, tempRect[2]+30))
		new_number[15:tempRect[3]+15, 15:tempRect[2]+15] = number
		new_number = cv2.erode(new_number, kernel, iterations=3)
		#new_number = cv2.subtract(255, new_number)
		numbers.append(new_number)
		j += 1

test_data = []
cv2.imshow('Threshold', binary)

for i, number in enumerate(numbers):
	number = cv2.resize(number, (28, 28))
	cv2.imshow('Roi {}'.format(i), number)
	test_data.append(np.reshape(number, (784,1)))

np.savetxt('test.csv', test_data)

net = OCR_NN.Network([784, 100, 10])
results = net.test()

print(results)

for i, result in enumerate(results):
	cv2.putText(output, str(result), (rect[i][0], rect[i][1] + rect[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imshow('Result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
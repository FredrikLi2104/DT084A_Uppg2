import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def initialize_camera(camera_index=0):
	cap = cv2.VideoCapture(camera_index)
    
	# Check if the cam is opened correctly
	if not cap.isOpened():
		print("Error: Could not open camera.")
		return None
        
	return cap

	

def high_pass_filter(img):
	
	# Compute the 2 dimensional discrete Fourier Transform
	G = np.fft.fft2(img)
	# Shifts the zero frequency component to the center of the spectrum.
	# If array = [0, 1, 2, 3, 4, 5, 6, 7] then after shift it is
	# [-3, -2, -1, 0, 1, 2, 3]
	G_shift = np.fft.fftshift(G)

	# M is nr of tuples in img and N is the index of the tuples.
	M, N = img.shape
	
	D0 = 30
	# Creates an array of evenly spaced values of the tuples with their corresponding
	# values, 1 step between all starting at index 0.
	u = np.arange(M)
	v = np.arange(N)

	# Creates a coordinate matrice of the arrays v and u
	U, V = np.meshgrid(v, u)

	# (U - N/2 and V - M/2) shifts the origin to the center of the image.
	# np.sqrt(...) calculates the Euclidean distance of each frquency component
	# from the center.
	# Low frequencies are near the center and high frequencies are farther from the center
	# The distance D helps define which frequencies to keep or remove. 
	D = np.sqrt((U - N/2)**2 + (V - M/2)**2)

	# D > D0 keeps only the frequencies higher than specified D0 according to D.
	# This in order to keep high frequencies as per high pass filter.
	H = np.float32(D > D0)

	

	# Multiply the transformed image G_shift with the hgh pass filter H
	G_shift_filtered = G_shift * H
	
	# Shifts the frequency spectrum back to its original positioning.
	# fftshift was done earlier to move lower frequencies to the center
	# for easier filtering, ifftshift moves them back.
	f_ishift = np.fft.ifftshift(G_shift_filtered)
	# ifft2 inverse Fast Fourier transforms it back to an image.
	img_back = np.fft.ifft2(f_ishift)
	# takes only the magnitude of the result ensuring valid pixel values.
	img_back = np.abs(img_back)
	# normalise scales the image values between black and white to ensure
	# clearer visualization.
	img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
	return np.uint8(img_back)

def process_frame(frame):
	# Add your processing code here
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Define range of blue color in HSV
	lower_green = np.array([110,50,50])
	upper_green = np.array([130, 255, 255])
	
	# Threshold the HSV image to get only blue colors
	frame = cv2.inRange(hsv, lower_green, upper_green)
	return frame

def main():
	# Initialize the camera
	cap = initialize_camera()
	if cap is None:
		sys.exit(1)

	
	BLUE = [255,0,0]
	print("Camera feed started. Press 'q' to quit.")



	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		H = high_pass_filter(gray_frame)

		# Check if frame is captured successfully
		if not ret:
			print("Error: Can't receive frame from camera.")
			break

		constant= cv2.copyMakeBorder(frame,300,300,300,300,cv2.BORDER_REFLECT)
		cv2.imshow('Filter', H)
		# Process the frame with a chosen (set) of functions
		#output_frame = process_frame(frame)
        
		# Display the original frame
		cv2.imshow('Original Frame', frame)

		# Display original frame with blue border
		cv2.imshow('Olle', constant)

		# Display the processed frame
		#cv2.imshow('Processed Frame', output_frame)

		# Check for 'q' key press to quit the application
		# waitKey(1) returns -1 if no key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main() 

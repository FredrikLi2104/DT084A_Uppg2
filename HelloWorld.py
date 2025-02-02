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

	

def low_pass_filter(img):
	# image in frequency domain
	F = np.fft.fft2(img)

	Fshift = np.fft.fftshift(F)
	plt.imshow(np.log1p(np.abs(F)), cmap = 'gray')
	plt.axis('off')
	plt.show()



	M,N = img.shape
	H = np.zeros((M,N), dtype=np.float32)
	D0 = 50
	for u in range(M):
		for v in range(N):
			D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
			if D <= D0:
				H[u,v] = 1
			else:
				H[u,v] = 0

	plt.imshow(H, cmap='gray')
	plt.axis('off')
	plt.show()

	# Ideal Low Pass Filter
	Gshift = Fshift * H
	G = np.fft.ifftshift(Gshift)
	plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
	plt.axis('off')
	plt.show()

	g = np.abs(np.fft.ifft2(G))
	plt.imshow(g, cmap='gray')
	plt.axis('off')
	plt.show()


	# High Pass Filter
	H = 1 - H

	plt.imshow(H, cmap='gray')
	plt.axis('off')
	plt.show()


	# Ideal High Pass Filter
	Gshift = Fshift * H
	plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
	plt.axis('off')
	plt.show()


	# Inverse Fourier Transform
	G = np.fft.ifftshift(Gshift)
	plt.imshow(np.log1p(np.abs(G)), cmap='gray')
	plt.axis('off')
	plt.show()

	g = np.abs(np.fft.ifft2(G))
	plt.imshow(g, cmap='gray')
	plt.axis('off')
	plt.show()

	return H

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

	f = cv2.imread('cube.jpg', 0)
	
	H = low_pass_filter(f)



	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()
		

		# Check if frame is captured successfully
		if not ret:
			print("Error: Can't receive frame from camera.")
			break

		constant= cv2.copyMakeBorder(frame,300,300,300,300,cv2.BORDER_REFLECT)

		# Process the frame with a chosen (set) of functions
		output_frame = process_frame(frame)
        
		# Display the original frame
		cv2.imshow('Original Frame', frame)

		# Display original frame with blue border
		cv2.imshow('Olle', constant)

		# Display the processed frame
		cv2.imshow('Processed Frame', output_frame)

		# Check for 'q' key press to quit the application
		# waitKey(1) returns -1 if no key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Clean up
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main() 
import cv2
import numpy as np
from matplotlib import pyplot as plt

def filter(image, cutoff_freq):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Perform Fourier Transform
    f_transform = np.fft.fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)

    # Create a mask for filtering
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1

    # Apply mask to Fourier-transformed image
    f_shift_filtered = f_shift * mask

    # Perform Inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

# Get user input for image path and cutoff frequency
image_path = input("Enter the path to the input image: ")
cutoff_frequency = int(input("Enter the cutoff frequency for the low-pass filter: "))

# Load the image
input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Apply filter
filtered_image = filter(input_image, cutoff_frequency)

# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
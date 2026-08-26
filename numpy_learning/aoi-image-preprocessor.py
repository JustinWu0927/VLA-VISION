from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# grayscale and turn into array
img = np.array(Image.open('/Users/wucheng-yan/Desktop/vla-project/dim-factory-image.jpg').convert('L'))

# histogram equalization
def histeq(img, nbr_bins=256):
    imhist, bins = np.histogram(img.flatten(), nbr_bins, density=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    img2 = np.interp(img.flatten(), bins[:-1], cdf)

    return img2.reshape(img.shape), cdf

img2, cdf = histeq(img)

# set up subplot
fig, axs = plt.subplots(2, 2, figsize=(12, 5))

# show original image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# show image2
axs[0, 1].imshow(img2, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[0, 1].axis('off')

# show original histogram
axs[1, 0].hist(img.flatten(), bins=128)
axs[1, 0].set_title('Original Histogram')
axs[1, 0].axis('off')

# show equalized histogram
axs[1, 1].hist(img2.flatten(), bins=128)
axs[1, 1].set_title('Equalized Histogram')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
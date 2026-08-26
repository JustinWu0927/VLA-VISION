from PIL import Image
from pylab import *

img = array(Image.open('/Users/wucheng-yan/Desktop/cv_screenshot.png').convert('L'))

figure()
img1 = 255 - img
imshow(img1, gray())

figure()
img2 = (100 / 255) * img + 100
imshow(img2, gray())

figure()
img3 = 255 * (img / 255) ** 2
imshow(img3, gray())

show()
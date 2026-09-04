from PIL import Image
from numpy import *
from pylab import *
from scipy.ndimage import filters

img1 = array(Image.open('/Users/wucheng-yan/Desktop/vla-project/new_wrench.jpg'))
img2 = array(Image.open('/Users/wucheng-yan/Desktop/vla-project/old_wrench.jpg'))

img3 = array(Image.open('/Users/wucheng-yan/Desktop/vla-project/new_wrench.jpg').convert('L'), 'f')
img4 = array(Image.open('/Users/wucheng-yan/Desktop/vla-project/old_wrench.jpg').convert('L'), 'f')

sigma = 5
img3 = filters.gaussian_filter(img3, sigma)
img4 = filters.gaussian_filter(img4, sigma)

diff_map = abs(img3 - img4)

threshold = 20
mark = diff_map > threshold

result = img2.copy()
result[mark] = [255, 0, 0]

figure()
subplot(1, 3, 1)
title('New Wrench')
imshow(img1)
axis('off')

subplot(1, 3, 2)
title('Old Wrench')
imshow(img2)
axis('off')

subplot(1, 3, 3)
title('Difference Map')
imshow(result)
axis('off')

show()
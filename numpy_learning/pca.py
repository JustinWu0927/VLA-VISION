from PIL import Image
from numpy import *
from pylab import *
import pca
import pickle

img = array(Image.open(imglist[0]))
m ,n = img.shape[0:2]
imgnbr = len(imglist)

imgmat = array([array(Image.open(img).flatten()) for img in imglist], 'f')

v,s, immean = pca.pca(imgmat)

figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m, n))
for i in range(7):
    subplot(2,4,i+2)
    imshow(v[i].reshape(m, n))

show()

with open('eigenfaces.pkl', 'wb') as f:
    pickle.dump(immean, f)
    pickle.dump(v, f)
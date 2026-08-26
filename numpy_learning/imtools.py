from PIL import Image
import os

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

def imresize(img, sz):
    pil_img = Image.fromarray(uint8(img))
    return array(pil_img.resize(sz))

def histeq(img, nbr_bins = 256):
    imhist, bins = histogram(img.flatten(), nbr_bins, normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    im2 = interp(img.flatten(), bins[:-1], cdf)
    return im2.reshape(img.shape), cdf

def compute_average(imlist):
    averageimg = array(Image.open(imlist[0], 'f'))

    for imname in imlist[1:]:
        try:
            averageimg += aarray(Image.open(imname))

        except:
            print(imname + '...skipped')

    averageimg /= len(imlist)
    return uint8(averageimg)
    
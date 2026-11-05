import numpy as np
from PIL import ImageOps
from skimage.feature import hog
def features(im):
 im=ImageOps.fit(im.convert('RGB'),(96,96))
 arr=np.asarray(im)
 gray=np.asarray(im.convert('L'))
 h=hog(gray,orientations=9,pixels_per_cell=(12,12),cells_per_block=(2,2),feature_vector=True)
 hist=np.concatenate([np.histogram(arr[:,:,c],bins=16,range=(0,256),density=True)[0] for c in range(3)])
 return np.concatenate([h,hist]).astype('float32')

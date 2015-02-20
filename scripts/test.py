import cv2
import numpy as np
from matplotlib import pyplot as plt


perfect = cv2.imread('perfect.jpg',0)
ususable = cv2.imread('unusable.jpg',0)
faulty = cv2.imread('faulty.jpg',0)
img2 = perfect.copy()

#img = cv2.imread('1.jpg',0)
roi_faulty = faulty[160:250, 300:390]
roi_ususable = ususable[160:250, 300:390]
roi_perfect =perfect[160:250, 300:390] 

template = roi_faulty
w, h = template.shape[::-1]

"""
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

"""


edges_faulty = cv2.Canny(roi_faulty,50,50)
edges_ususable = cv2.Canny(roi_ususable,50,50)
edges_perfect = cv2.Canny(roi_perfect,50,50)

plt.subplot(131),plt.imshow(edges_perfect,cmap = 'gray')
plt.title('Edges Perfect'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges_ususable,cmap = 'gray')
plt.title('Edges Usable'), plt.xticks([]), plt.yticks([])

#plt.subplot(132),plt.imshow(faulty[120:300, 250:450], cmap='gray')
plt.subplot(133),plt.imshow(edges_faulty,cmap = 'gray')
plt.title('Edges Faulty'), plt.xticks([]), plt.yticks([])

#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

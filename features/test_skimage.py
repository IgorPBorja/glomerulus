from skimage import data
from skimage import filters
from skimage import feature
import matplotlib.pyplot as plt
import config

camera = data.camera()
print(f"Original image is of datatype {camera.dtype}")

edges = filters.sobel(camera)
print(f"Image post-sobel is of datatype {edges.dtype}")

lbp = lambda img: feature.local_binary_pattern(
            img, 
            P = config.P, 
            R = config.R, 
            method="default"
        )

lbp_img = lbp(camera)
print(f"Image post-sobel is of datatype {lbp_img.dtype}")
print(f"Max of LBP={lbp_img.max()} | Min of LBP={lbp_img.min()}")

glcm = lambda img: feature.graycomatrix(
            image=img,
            distances=config.DISTANCES,
            angles=config.ANGLES
        )
glcm_img = glcm(camera)
print(f"Image post-glcm is of datatype {glcm_img.dtype}")
print(f"Max of LBP={glcm_img.max()} | Min of LBP={glcm_img.min()}")

fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(camera, cmap='gray')
ax[0][0].axis('off')
ax[0][1].imshow(edges, cmap='gray')
ax[0][1].axis('off')
ax[1][0].imshow(lbp_img, cmap='gray')
ax[1][0].axis('off')
ax[1][1].imshow(glcm_img[:, :, 0, 0], cmap='gray')
ax[1][1].axis('off')
plt.show()

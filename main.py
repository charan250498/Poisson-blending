from pickletools import uint8
from turtle import window_height
import cv2
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt

from align_target import align_target

from matrix_operations import compute_A_matrix, compute_different_A_matrix, get_A_matrix
from matrix_operations import compute_b_matrix, compute_different_b_matrix, get_b_matrix

def poisson_blend(source_image, target_image, target_mask):
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    #target_mask = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    
    height, width, _ = source_image.shape
    k = height*width

    print("height: "+str(height)+" width: "+str(width))

    final_output = np.zeros((height, width, 3))
    least_square_error = 0
    for i in range(3):
        A = get_A_matrix(height, width, target_mask).tocsc()
        b = get_b_matrix(target_image[:,:,i], source_image[:,:,i], target_mask, height, width)
        output_image = sparse.linalg.spsolve(A, b)
        output_image = output_image.reshape(height, width)
        final_output[:,:,i] = np.copy(output_image)

        # least square calculation
        Av = A.dot(final_output[:,:,i].reshape(height*width))
        least_square_error += np.linalg.norm(Av-b)


    final_output = final_output.astype(int)

    print("Least Square Error: ",least_square_error/3)

    plt.imshow(final_output)
    plt.pause(7)


if __name__ == '__main__':
    #read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)
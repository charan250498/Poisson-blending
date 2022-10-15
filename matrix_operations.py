import numpy as np
from scipy import sparse

def compute_A_matrix(height, width):
    k = height*width
    row = np.array([])
    col = np.array([])
    data = np.array([])
    # Logic to fill inner square of coefficients
    for i in range(1, height-1):
        for j in range(1, width-1):
            row = np.concatenate((row, np.array([i*width+j, i*width+j, i*width+j, i*width+j, i*width+j])))
            col = np.concatenate((col, np.array([(i-1)*width+j, i*width-1+j, i*width+j, i*width+1+j, (i+1)*width+j])))
            data = np.concatenate((data, np.array([1, 1, -4, 1, 1])))

    # logic to fill the corners
    # [Left top corner, Right Top corner, Left bottom corner, Right Bottom corner]
    row = np.concatenate((row, np.array([0, width-1, (width-1)*height, k-1])))
    col = np.concatenate((col, np.array([0, width-1, (width-1)*height, k-1])))
    data = np.concatenate((data, np.array([1, 1, 1, 1])))
    
    # Logic to fill the edges
    # top edge
    for i in range(1, width-1):
        row = np.concatenate((row, np.array([i, i, i])))
        col = np.concatenate((col, np.array([i-1, i, i+1])))
        data = np.concatenate((data, np.array([1, -2, 1])))

    #bottom edge
    for i in range((k-1)-(width)+2, k-1):
        row = np.concatenate((row, np.array([i, i, i])))
        col = np.concatenate((col, np.array([i-1, i, i+1])))
        data = np.concatenate((data, np.array([1, -2, 1])))
        
    #left edge
    for i in range(1, height-1):
        row = np.concatenate((row, np.array([i*width, i*width, i*width])))
        col = np.concatenate((col, np.array([(i-1)*width, i*width, (i+1)*width])))
        data = np.concatenate((data, np.array([1, -2, 1])))

    #right edge
    for i in range(1, height-1):
        row = np.concatenate((row, np.array([(i+1)*width-1, (i+1)*width-1, (i+1)*width-1])))
        col = np.concatenate((col, np.array([(i)*width-1, (i+1)*width-1, (i+2)*width-1])))
        data = np.concatenate((data, np.array([1, -2, 1])))
    
    return row, col, data

def compute_different_A_matrix(height, width, mask):
    A = sparse.lil_matrix((height*width, height*width))
    mask = mask.reshape(height*width)
    #Contruct laplacian here with required info to inpaint.
    for i in range(height*width):
        if mask[i] > 0:
            A[i,i] = 4
            A[i,i-1] = -1
            A[i,i+1] = -1
            A[i,i-width] = -1
            A[i,i+width] = -1
            #A[i,i] = 1
        else:
            A[i,i] = 1

    return A

def compute_b_matrix(target_image, source_image, mask, b, height, width):
    for i in range(height):
        for j in range(width):
            if mask[i,j] > 0:
                if (i==0 and j==0) or (i==height-1 and j==0) or (i==0 and j==width-1) or (i==height-1 and j==width-1):
                    b[i,j] = 35
                elif i==0 or i==height-1:
                    b[i,j] = (source_image[i, j+1] - source_image[i,j]) + (source_image[i,j] - source_image[i, j-1])
                elif j==0 or j==width-1:
                    b[i,j] = (source_image[i-1,j] - source_image[i,j]) + (source_image[i,j] - source_image[i+1,j])
                else:
                    b[i,j] = (source_image[i, j+1] - source_image[i,j]) + (source_image[i,j] - source_image[i, j-1]) + (source_image[i-1,j] - source_image[i,j]) + (source_image[i,j] - source_image[i+1,j])
            else:
                b[i,j] = target_image[i, j]

def compute_different_b_matrix(target_image, source_image, mask, height, width):
    b = np.zeros(height*width, dtype='float32')
    mask = mask.flatten(order='C')
    t_image = target_image.flatten(order='C')
    s_image = source_image.flatten(order='C')
    for i in range(height*width):
        if mask[i] > 0:
            #if (i%width != 0):
            s_left = s_image[i] - s_image[i-1]
            #else:
            #    s_left = s_image[i]
            #if (i % width != width-1):
            s_right = s_image[i] - s_image[i+1]
            #else:
            #    s_right = s_image[i]
            #if (i-width > 0):
            s_up = s_image[i] - s_image[i-width]
            #else:
            #    s_up = s_image[i]
            #if(i+width < height*width):
            s_down = s_image[i] - s_image[i+width]
            #else:
            #    s_down = s_image[i]
            b[i] = s_left + s_right + s_up + s_down
        #    b[i] = t_image[i]
        else:
            b[i] = t_image[i]
    return b

def get_A_matrix(height, width, mask):
    A = sparse.lil_matrix((height*width, height*width))
    mask = mask.reshape(height*width)
    #Contruct laplacian here with required info to inpaint.
    for i in range(height*width):
        if mask[i] > 0:
            if mask[i-1] == 0 or mask[i+1] == 0 or mask[i-width] == 0 or mask[i+width] == 0:
                A[i,i] = 1
            else :
                A[i,i] = 4
                A[i,i-1] = -1
                A[i,i+1] = -1
                A[i,i-width] = -1
                A[i,i+width] = -1
        else:
            A[i,i] = 1
    return A

def get_b_matrix(target_image, source_image, mask, height, width):
    b = np.zeros(height*width)
    for i in range(height):
        for j in range(width):
            if mask[i,j] > 0:
                if mask[i-1,j] == 0 or mask[i+1, j] == 0 or mask[i,j-1] == 0 or mask[i, j+1] == 0:
                    b[i*width+j] = target_image[i,j]
                else:
                    b[i*width+j] = 4*source_image[i,j] - source_image[i-1,j] - source_image[i+1,j] - source_image[i,j-1] - source_image[i,j+1]
            else:
                b[i*width+j] = target_image[i,j]
    return b
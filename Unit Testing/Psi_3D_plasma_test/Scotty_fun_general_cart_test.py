# -*- coding: utf-8 -*-
"""
Created: 08/07/2022
Last Updated: 02/08/2022
@author: Tan Zheng Yang

THIS IS A COPY SCRIPT FOR TESTING. DO NOT EDIT CODE HERE, EDIT IN THE ORIGINAL 
FOLDER.
"""

import numpy as np
from scipy import constants as constants
from scipy import interpolate as interpolate
from scipy import optimize as optimize
from scipy import integrate as integrate
import sys





########## STUFF TO MAKE LIFE EASIER ##########

def read_floats_into_list_until(terminator, lines):
    """
    DESCRIPTION
    ==============================
    Read each line of a text file through continuous application of the 
    readline() function using a while loop. The while loop terminates when 
    a particular 'terminator' string is read or an empty string '' is read.
      
    If the loop does not break, then the line in the file is separated into
    individual floats via the split() function. However, as the floats are read 
    as strings since they are obtained from a text file, application of the 
    map() and float() functions are required.
    
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    The 'try-except' block was removed since StopIteration only occurs when
    using the next() function. When using readline(), if there is no next
    line, readline() will produce an empty string.
    
    INPUT
    ==============================
    terminator (string): 
        the particular string that breaks the while loop when it is read.
        
        
    lines (text file): 
        text file to be read. Note that if the text file is 'test.txt', then
        you MUST first write:
            
            file = open('test.txt', 'r')
            
        you then pass the above variable "file" as the argument 'lines'.
        
    OUTPUT
    ==============================
    lst: 
        Produces the list of floats read up to before the terminator string is
        read.
    """
    
    lst = []
    while True:
        
        line = lines.readline()
        
        if terminator in line: # 'terminator' string found in file
            # For debugging
            # print('terminator found!')
            break 
        elif not line: # empty string found in file
            # For debugging
            # print('empty string found!')
            break
        else:
            # Converts the floats (read as strings from text file) into actual
            # Python floats
            print(line)
            lst.extend(map(float, line.split())) 
    return lst


# For debugging of function read_floats_into_list_until
#test1 = open('tests/test1.txt', 'r')
#print(read_floats_into_list_until('2', test1))


def find_nearest(array, value):
    
    """
    DESCRIPTION
    ==============================
    Finds the FIRST index of the value NEAREST to the desired "value" (float)
    in "array". Does so by first subtracting the desired float from all values
    in the array. Then, the value closest to the desired "value" will be
    the smallest (it will be theoretically zero if it is exactly equal).
    Finally, an argmin() function can be applied to find the index of this 
    nearest value.
    
    Note that there may be multiple values in the array that are equal leading
    to in fact, multiple indexes of items in the array that are the "nearest".
    This function simply picks the first one.
    
    INPUT 
    ==============================
    array (numpy array):
        a numpy array containing various values.
        
    value (float):
        the float desired in which we would like to find the nearest value in
        the array.
        
    OUTPUT
    ==============================
    idx (integer):
        index of "value" in "array".
    """
    
    # converts data into a numpy array
    array = np.asarray(array) 
    
    idx = int(np.argmin((np.abs(array - value))))
    return idx
    

# For debugging of function find_nearest
#test_lst = [1.7, 3.2, 6.6, 7.7, 6.6]
#print(find_nearest(test_lst, 1.2))


def contract_special(arg_a,arg_b): 
    
    """
    DESCRIPTION
    ==============================
    Note that in our descriptions below, matrices are rank 3 tensors while 
    vectors are rank 2 tensors. This is because the first index always
    corresponds to a temporal index. For example, we consider the following to
    be a vector:
        
        A = [[1, 2, 3], [4, 5, 6]]
        
    which implies that A = [1, 2, 3] at time t_0 and A = [4, 5, 6] at time t_1.
    
    Now, we define arg_a = A and arg_b = B. This function performs a 
    contraction for the following cases:
        
        (a) A is a matrix T*M*N and B is a vector T*N. For each 
            T (each instance of time), we contract the N index. That is, we 
            are performing the tensor contraction:
                
                C_{ijm} = A_{ijk}B_{mk} 
                
            where the spatial indexes are:
                1 < j < M
                1 < k < N
            and the temporal indexes are:
                1 < i < T
                1 < m < T
                
            Notice that the resultant tensor C_{ijm} has two temporal indexes
            which does not make sense. Therefore, it only makes sense if the
            two temporal indexes are the same, that is, we only consider:
                
                D_{mj} = C_{mjm} 
                
            in order to have a valid physical interpretation. Note that 
            C_{mjm} implies that we have set m = i. It does NOT imply a sum
            over the index m.
        
        (b) A is a vector T*M and B is a matrix T*M*N. For each 
            T (each instance of time), we contract the M index. That is, we 
            are performing the tensor contraction:
                
                C_{ikm} = A_{mj}B_{ijk} = B_{ijk}A_{mj}
                
            (we ALWAYS put the matrix on the left) where the spatial indexes 
            are:
                1 < j < M
                1 < k < N
            and the temporal indexes are:
                1 < m < T
                1 < i < T
                
            Notice that the resultant tensor C_{ikm} has two temporal indexes
            which does not make sense. Therefore, it only makes sense if the
            two temporal indexes are the same, that is, we only consider:
                
                D_{mk} = C_{mkm}
                
            in order to have a valid physical interpretation. Note that 
            C_{mkm} implies that we have set i = m. It does NOT imply a sum 
            over the index m.
        
        (c) A is a vector T*N and B is a vector T*N. For each
            T (each instance of time), we contract the N index. That is, we 
            are performing the tensor contraction:
                
                C_{mi} = A_{mj}B_{ij}
                
            where the spatial index is:
                1 < j < N
            and the temporal indexes are:
                1 < m < T
                1 < i < T
                
            Notice that the resultant tensor C_{mi} has two temporal indexes
            m and i which does not make sense. Therefore, it only makes sense 
            if the two temporal indexes are the same, that is, we only 
            consider:
                
                D_{i} = C_{ii}
                
            in order to have a valid physical interpretation. Note that C_{ii}
            implies that we have set i = m. It does NOT imply a sum over 
            the index i.
    
    Note that while we wrote here that the first index is 1, it is 0 in Python
    and hence in our code.

    Valerian introduces this to provide some efficient contraction procedures.
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Computed the tensor rank of arg_a and arg_b earlier instead of in the 
      "if" statements.
    
    - For CASE(b), changed the argument of np.diagonal from 
      "axis1 = 0, axis2 = 2" to "axis1 = 0, axis2 = 1" since we should be
      summing over the first and second indexes (temporal) of C_{mik}
    
    INPUT
    ==============================
    arg_a (numpy array):
        the first tensor.
    
    arg_b (numpy array):
        the second tensor.
        
    OUTPUT
    ==============================
    result (numpy array):
        the resulting contracted tensor.
    """
    
    # Calculates the TENSOR rank of arg_a and arg_b, this is equivalent to
    # the number of dimensions of the array in the data structure sense 
    # (NOT the mathematical sense, which is the matrix rank).
    # 
    # The number of dimensions in the data structure sense is equivalent to the
    # number of scalar indices required to obtain a scalar value. For example,
    # if arg_a[1][2][3] gives a scalar value, then arg_a has a dimension of 3.
    #
    # Note that np.ndim(arg_a) is equivalent to len(arg_a.shape)
    arg_a_rank = np.ndim(arg_a)
    arg_b_rank = np.ndim(arg_b)
    
    # CASE (a): 
    # A = arg_a is a matrix (rank 3 tensor as there is a temporal index)
    # B = arg_b is a vector (rank 2 tensor as there is a temporal index)
    if (arg_a_rank == 3 and arg_b_rank == 2):
        
        # Always label the matrix and vector because we do np.tensordot()
        # with the matrix on the left
        matrix = arg_a
        vector = arg_b
        
        # Performs the contraction C_{ijm} = A_{ijk}B_{mk} 
        #
        # "axes = ((2), (1))" indicates the axes to be sum over. In this case,
        # we are summing over the third index (Python index 2) of A = arg_a 
        # and the second index (Python index 1) of B = arg_b
        intermediate_result = np.tensordot(matrix, vector, axes = ((2), (1)))
        
        # Now, the tensor C_{ijm} has two temporal indexes i and m.
        # This has no physical sense since there shouldn't be two directions
        # of time. Therefore, it is only logical to consider the tensor:
        #   D_{mj} = C_{mjm}
        # where we have set i = m for the tensor C_{ijm}
        #
        # Since we want the first and last index to be the same in C_{mjm},
        # these are specified by the argument "axis1 = 0" and "axis2 = 2" 
        # respectively. 
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=2)
        
        # The above result is now of the form [C_{1j1}, C_{2j2}, ...]
        # in other words, each COLUMN corresponds to the same time.
        #
        # This is not what we want as our initial input has the ROWS
        # that correspond to the same time. Therefore, we must perform a
        # transpose
        result = np.transpose(result)
        
        
    # CASE (b):
    # A = arg_a is a vector (rank 2 tensor as there is a temporal index)
    # B = arg_b is a matrix (rank 3 tensor as there is a temporal index)  
    elif (arg_a_rank == 2 and arg_b_rank == 3):
        
        # Label matrix and vector
        matrix = arg_b 
        vector = arg_a 
        
        # Performs the contraction C_{ikm} = B_{ijk}A_{mj}
        #
        # Remember that np.tensordot() is always done with the matrix on the
        # LEFT
        #
        # "axes = ((1), (1))" indicates the axes to be sum over. In this case,
        # we are summing over the second index (Python index 1) of B = arg_b 
        # and the second index (Python index 1) of B = arg_b
        intermediate_result = np.tensordot(matrix, vector, axes = ((1), (1)))
        
        # Similar to CASE (a), the tensor C_{ikm} has two temporal indexes
        # m and i. So we need to set m = i. Since we want the first and third
        # index to be the same in C_{ikm}, these are specified by the argument
        # "axis1 = 0" and "axis2 = 2" respectively.
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=2)
        
        # Similar to CASE (a), need to transpose
        result = np.transpose(result)
        
    # CASE (c): 
    # A = arg_a is a vector (rank 2 tensor as there is a temporal index)
    # B = arg_b is a vector (rank 2 tensor as there is a temporal index) 
    elif (arg_a_rank == 2 and arg_b_rank == 2):
        
        # Both are vectors
        vector1 = arg_a
        vector2 = arg_b
        
        # Performs the contraction C_{mi} = A_{mj}B_{ij}
        #
        # "axes = ((1), (1))" indicates the axes to be sum over. In this case,
        # we are summing over the second index (Python index 1) of A = arg_a 
        # and the second index (Python index 1) of B = arg_b
        intermediate_result = np.tensordot(vector1,vector2, ((1), (1)))
        
        # Similar to CASE (a), then tensor C_{mi} has two temporal indexes 
        # m and i. So we need to set m = i. Since we want the first and second
        # index to be the same in C_{mi}, these are specified by the argument
        # "axis1 = 0" and "axis2 = 1" respectively.
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=1)
        
        # Similar to CASE (a), need to transpose
        result = np.transpose(result)
        
    # If not CASE (a), (b) or (c), then we are dealing with tensors of the
    # wrong dimension:
    else:
        print('Error: Invalid dimensions')
    return result
        
        
def make_unit_vector_from_cross_product(vector_a, vector_b):
    """
    DESCRIPTION
    ==============================
    For two vectors A = vector_a and B = vector_b, compute the cross product
    and normalize the result. Note that the vectors A and B are represented
    by TWO indexes, with the first index (row index) being that for time. 
    For example, we may write:
        
        A = [[0, 1, 2], [3, 4, 5]]
        
    meaning that A = [0, 1, 2] at t_0 and A = [3, 4, 5] at t_1. Therefore
    if A and B have multiple rows (the first index is not just one), then we
    are doing the cross product of A and B at multiple times. For example,
    if:
        
        A = [[0, 1, 2], [3, 4, 5], [6, 7, 8], ...]
        B = [[9, 10, 11], [12, 13, 14], [15, 16, 17], ...]
        
    which means that at t_0, the cross product is between A = [0, 1, 2] and
    B = [9, 10, 11]. At t_1, it is between A = [3, 4, 5] and B = [12, 13, 14]
    and so on.
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Replaced "axis = -1" with "axis = 1" from the np.linalg.norm() function 
      since traditionally, we should only be using "axis = 0" or "axis = 1".
      The former computes the norms column-wise while the latter computes
      the norms row-wise. For example if:
          
          A = [[1, 2], [3, 4]]
          
      then "axis = 0" results in the norm being \sqrt{1^2 + 3^2} and
      \sqrt{2^2 + 4^2}. On the other hand, "axis = 1" results in the norms
      being \sqrt{1^2 + 2^2} and \sqrt{3^2 + 4^2}
      
     - Replaced ".T" with "np.transpose" to make it more obvious.
    
    INPUT
    ==============================
    vector_a (numpy array):
        vector with the condition that np.shape(vector_a) = (n,3). The first
        index (number of rows) is the temporal index while the second index
        (numer of columns) is the spatial index, which is always 3 since we are
        working in three dimensions.
        
    vector_b (numpy array):
        vector with similar conditions imposed as in vector_a. However, it can 
        also take on the value of a constant vector where 
        np.shape(vector_b) = (3,)
        
    OUTPUT
    ==============================
    output_unit_vector (numpy array):
        the resultant normalised unit vector after the cross product AxB. 
        If A or B had multiple rows, then the output is a matrix also with
        multiple rows, with each row indicating the cross product result at
        different times.
    """
    
    # Computes the cross product between the vectors
    output_vector = np.cross(vector_a, vector_b)
    
    # Computes the magnitude of the resultant cross products
    #
    # "axis = 1" means the magnitudes are being computed row-wise
    #
    # Hence, if output_vector_magnitude is a 1D array, each element corresponds
    # to the norm of each row starting from the top of output_vector
    output_vector_magnitude = np.linalg.norm(output_vector, axis = 1)
    
    # output_vector_magnitude is a 1D array (row vector) where each column
    # represents the magnitude of each row vector (vector at different times)
    # of output_vector.
    #
    # To perform normalization, we need to divide each row in output_vector
    # by its corresponding magnitude value in output_vector_magnitude. For
    # example, the first row elements of output_vector must be divided by the
    # first column of output_vector_magnitude, the second row of output_vector
    # must be divided by the second column of output_vector_magnitude and so
    # on. (clearly, some sort of transpose operation is needed)
    #
    # Therefore, the first step is to use 
    # np.tile(output_vector_magnitude, (3,1))
    # to obtain the matrix:
    #
    # [
    #  output_vector_magnitude,
    #  output_vector_magnitude
    #  output_vector_magnitude
    # ]
    #
    # In other words, we repeat the array output_vector_magnitude three times
    # row-wise.
    #
    # Lastly, because np.shape(output_vector) = (n, 3), we need to tranpose 
    # the above matrix to ensure the correct elements in output_vector are
    # being divided by the correct corresponding magnitudes.
    denominator = np.transpose(np.tile(output_vector_magnitude, (3,1)))
    
    # Perform element wise division to obtain the correct unit vector for all
    # times (row indexes)
    output_unit_vector = output_vector / denominator
    
    return output_unit_vector


# For debugging of function make_unit_vector_from_cross_product
#test_vec = [[1, 5, 3], [2, 7, 1], [4, 4, 9]]
#test_vec_2 = [[3, 5, 5], [1, 9, 2], [0, 0, 1]]
#test_vec_2_constant = [3, 3, 4]
#print(make_unit_vector_from_cross_product(test_vec, test_vec_2))
#print(make_unit_vector_from_cross_product(test_vec, test_vec_2_constant))


def find_inverse_2D(matrix_2D):
    """
    DESCRIPTION
    ==============================
    Finds the inverse of a 2D matrix using cofactor matrix. Much faster than
    np.linalg.inv()
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Computed the cofactor matrix first to make the code more readable
    
    INPUT
    ==============================
    matrix_2D (numpy array):
        matrix to be inverted.
    
    OUTPUT
    ==============================
    matrix_2D_inverse (numpy array):
        the inverted matrix.
    """    
    
    # Create an empty 2x2 matrix
    # "complex128" indicates that the values in the matrix are complex
    # and requires 128bits of memory (64bits for the real part and
    # 64 bits for the complex part). We do not use 32bits for the
    # real/complex number as the maximum integer is 2147483647
    # (31 bits for the number and 1 more bit for the sign)
    matrix_2D_inverse = np.zeros([2,2], dtype = "complex128")
    
    # Calculate the determinant
    determinant = (matrix_2D[0,0]*matrix_2D[1,1]
                       - matrix_2D[0,1]*matrix_2D[1,0]
                    )
    
    # Calculate the cofactor matrix
    matrix_2D_inverse[0, 0] = matrix_2D[1,1]
    matrix_2D_inverse[0, 1] = -matrix_2D[1,0]
    matrix_2D_inverse[1, 0] = -matrix_2D[0,1]
    matrix_2D_inverse[1, 1] = matrix_2D[0,0]
    
    # Calculate the inverse
    matrix_2D_inverse = (1/determinant) * matrix_2D_inverse
    return matrix_2D_inverse


# For debugging of function find_inverse_2D
#test = np.array([[4, 7], [2, 6]])
#print(find_inverse_2D(test))
#print(find_inverse_2Ds(test))


def find_x0(xs,ys,y0): 
    """
    DESCRIPTION
    ==============================
    Given a value y0 on a curve specified by the points xs and ys, finds the
    corresponding x0 on the curve. Does so by:
        1. Finds the nearest y-value in ys 
        2. Creates a fine mesh in the neighbourhood of the nearest y-value
        3. Utilizes an interpolating function to find an even more accurate
           value of the nearest y-value to y0
        4. Finds the corresponding x0
        
     MODIFICATIONS FROM VALERIANS ORIGINAL CODE
     ==============================
     - Shifted the creation of the interpolating function to later to make the
       flow of the code more logical.
    
    INPUT
    ==============================
    xs (numpy array):
        list of x coordinates of the curve we wish to interpolate.
        
    ys (numpy array):
        list of the y coordinates, corresponding to xs, of the curve we wish
        to interpolate.
        
    y0 (float):
        y coordinate of which we would like to use interpolation to find the
        corresponding x coordinate.
    
    OUTPUT
    ==============================
    x0 (float):
        the corresponding x coordinate to y0 after interpolation.
    
    """
    # finds the index of the value nearest to y0 in ys
    index_guess = find_nearest(ys,y0)
    
    ### CREATION OF FINE MESH ###
    #
    # CASE 1: nearest value in ys to y0 is the LEFT most point of the 
    #         range of ys
    if index_guess == 0:
        
        # Divide the x-axis interval between the left most and second left most
        # point in xs into 100 equally spaced intervals (101 points).
        #
        # We will later then interpolate these 101 points on the x-axis to 
        # obtain the corresponding points on the y-axis, and then apply the
        # find_nearest function again to find a more accurate y value nearer to
        # the actual value of y0
        xs_fine = np.linspace(xs[0],xs[index_guess+1],101)
        
    # CASE 2: nearest value in ys to y0 is the RIGHT most point of the 
    #         range of ys
    elif index_guess == len(xs)-1:
        
        # Divide the x-axis interval between the right most and second right
        # most point in xs into 100 equally spaced intervals (101 points).
        #
        # We will later then interpolate these 101 points on the x-axis to 
        # obtain the corresponding points on the y-axis, and then apply the
        # find_nearest function again to find a more accurate y value nearer to
        # the actual value of y0
        xs_fine = np.linspace(xs[index_guess-1],xs[-1],101)
        
    # CASE 3: nearest value in ys to y0 is an interior point of ys
    else:
        
        # Divide the x-axis interval between the point before and after the
        # nearest value into 100 equally spaced intervals (101 points).
        #
        # We will later then interpolate these 101 points on the x-axis to 
        # obtain the corresponding points on the y-axis, and then apply the
        # find_nearest function again to find a more accurate y value nearer to
        # the actual value of y0
        xs_fine = np.linspace(xs[index_guess-1],xs[index_guess+1],101)
    
    
    # creates an 1D interpolation function using linear splines
    #
    # Note that by setting 'bounds_error = False', we are allowing for 
    # y0 to be a number OUTSIDE the range of ys. For y0 outside the range of
    # ys, the interpolation automatically assigns them the 'fill_value'
    # which in this case, the points will be automatically extrapolated.
    #
    # In case you are wondering, the method of extrapolation is also linear,
    # since we had specified kind='linear'. For example, for values much larger 
    # then the range, the function constructs a straight line from the last and
    # second last tuples of (xs, ys) and proceeds to do the extrapolation.
    interp_y = interpolate.interp1d(xs, ys,
                                    kind='linear', axis=-1, copy=True, 
                                    bounds_error=False,
                                    fill_value='extrapolate', 
                                    assume_sorted=False) 
    
    # interpolation to find the list of new y coordinates based on the new
    # fine mesh
    ys_fine = interp_y(xs_fine)
    
    # find the even more accurate value of y closest to y0
    index = find_nearest(ys_fine,y0)
    
    # find the corresponding value of x
    x0 = xs_fine[index]
    
    return x0

# For debugging of function find_x0
# xs = [-3, -2, -1, 0, 1, 2, 3]
# ys = [9, 4, 1, 0, 1, 4, 9]
# test_interp = interpolate.interp1d(xs, ys,
#                                 kind='linear', axis=-1, copy=True, 
#                                 bounds_error=False,
#                                 fill_value='extrapolate', 
#                                 assume_sorted=False)

# print(test_interp(-7))


def find_area_points(xs,ys,fraction_wanted): 
    """
    DESCRIPTION
    ==============================
    Given a curve specified by xs and ys, finds the points (x1, y1)
    and (x2, y2) on the left and right of the point of maximum ys, 
    (xs_max, ys_max), respectively, such that:
        
        total area under the curve (x1, y1) to (xs_max, ys_max)
        =
        total area under the curve (xs_max, ys_max) to (x2, y2)
        =
        fraction_wanted
        
    In other words, we are find thing points that are localized around
    (xs_max, ys_max) based on the area under the curve.
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Changed the calculation of the total area integration from
      "total_ys = integrate.simps(ys,xs)" to "total_ys = cumulative_ys[-1]"
      to be consistent with the method of integration.
      
    - In case 2, changed "(fraction_at_y_max + fraction_wanted) > 1.0" to 
      "fraction_wanted > 1.0 - fraction_at_y_max" to improve the code
      readability

    INPUT
    ==============================
    xs (numpy array):
        independent variable SORTED in ASCENDING order.
        
    ys (numpy array):
        dependent variable where ys[i] = f(xs[i]).
        
    fraction_wanted (float):
        fraction of the total area under the on EACH side of ys_max. That is,
        if fraction_wanted = 0.5, then we wish for half of the total area under
        the curve to be on the LEFT of ys_max and the other half of the total
        area to be on the RIGHT of ys_max.
    
    
    OUTPUT
    ==============================
    x_vals (numpy array):
        a 1x2 array such that x_vals[0] = x1 and x_vals[-1] = x2
        
    y_vals (numpy array):
        a 1x2 array such that y_vals[0] = y1 and y_vals[-1] = y2
    
    """
    
    # Checks that fraction_wanted is not outside the desired range of values
    #
    # For example, if fraction wanted = 0.6, then the fraction of the area 
    # under the curve from (x1, y1) to (xs_max, ys_max) will be 0.6 and 
    # the fraction of the area under the curve from (xs_max, ys_max) to 
    # (x2, y2) will also be 0.6. But this means that the total fractional area
    # from (x1, y1) to (x2, y2) would be 1.2 of the total area which 
    # obviously does not make any sense.
    if fraction_wanted > 0.5 or fraction_wanted < 0.0:
        print('Invalid value given for fraction')
        
    else:
        # Stores the coordinates (x1, y1) and (x2, y2)
        x_vals = np.zeros(2)
        y_vals = np.zeros(2)
        
        # Finds the index of the first occurence of the maximum value of ys 
        # in the array ys
        ys_max_idx = np.argmax(ys)
        
        # cumulative integration of ys as a function of xs using the
        # composite trapezoidal rule. In other words:
        #
        #   cumulative_ys[i] = total integral from xs[0] to xs[i]
        #
        # the "initial = 0" setting ensures that cumulative_ys[0] = 0
        cumulative_ys = integrate.cumulative_trapezoid(ys,xs,initial=0)
        
        # Finds the cumulative integration at from ys_0 to ys_max
        cumulative_ys_at_y_max = cumulative_ys[ys_max_idx]
        
        # Finds the total integration value
        total_ys = cumulative_ys[-1]
        
        # Calculates the fraction of the cumulative integral from ys_0 to 
        # ys_max over the total integral value
        fraction_at_y_max = cumulative_ys_at_y_max / total_ys
        
        # CASE 1: The area from ys_0 to ys_max (left side of the maximum point)
        #         is less than the fraction wanted. To deal with this, we 
        #         simply take (x1, y1) to be the (x[0], y[0]) and then, we find
        #         the point (x2, y2) such that the cumulative area from 
        #         (x1, y1) to (x2, y2) is 2*fraction wanted, which is the 
        #         total fractional area we want localized around 
        #         (xs_max, ys_max). This guarantess that (x1, y1) is LEFT of
        #         (xs_max, ys_max) and (x2, y2) is RIGHT of (xs_max, ys_max)
        if (fraction_at_y_max - fraction_wanted) < 0:
            
            # Helps us to find the point xs[i] such that the cumulative 
            # integral from xs[0] to xs[i] is zero. This will be (xs[0], ys[0])
            # = (x1, y1) and on the LEFT of (xs_max, ys_max)
            #
            # CHECK WITH VALERIAN WHY WE NEED TO DO INTERPOLATION?
            lower_fraction = 0.0
            
            # Helps us to find the point xs[j] such that the cumulative 
            # integral from xs[0] to xs[j] is twice the fraction wanted. 
            # This point, (xs[j], ys[j]) = (x2, y2) will be on the RIGHT of
            # (xs_max, ys_max)
            upper_fraction = 2 * fraction_wanted
            
        # CASE 2: The area from ys_max to ys[-1] ie. right of (xs_max, ys_max)
        #         until the right end point is less than the fraction wanted.
        #         To deal with this, we do what we did in case 1, except on the
        #         right end point instead of the left. We simply take 
        #         (x2, y2) to be the right end point (xs[-1], ys[-1]) and then,
        #         find the point (x1, y1) such that the area from 
        #         (xs[0], ys[0]) to (x1, y1) is 1 - 2*fraction_wanted, ensuring
        #         that the area from (x1, y1) to (x2, y2) is equivalent to 
        #         2*fraction_wanted which is what we desire. This guarantees
        #         that (x1, y1) is LEFT of (xs_max, ys_max) and (x2, y2)
        #         is RIGHT of (xs_max, ys_max)
        elif fraction_wanted > (1.0 - fraction_at_y_max) :
            
            # Helps us to find the point xs[j] such that the cumulative 
            # integral from xs[0] to xs[j] is 1. This will be (xs[-1], ys[-1])
            # = (x2, y2) and on the RIGHT of (xs_max, ys_max)
            #
            # CHECK WITH VALERIAN WHY WE NEED TO DO INTERPOLATION?
            upper_fraction = 1.0
            
            # Helps us to find the point xs[i] such that the cumulative 
            # integral from xs[0] to xs[i] is 1 - 2*fraction_wanted. This 
            # point, (xs[i], ys[i]) = (x1, y1) will be on the LEFT of 
            # (xs_max, ys_max) and ensures that the area from (x1, y1) to 
            # (x2, y2) is twice the fraction wanted, which is what we desire.
            lower_fraction = 1 - 2 * fraction_wanted
            
        # CASE 3: The fractional areas on the right and left of 
        #         (xs_max, ys_max) is greater than or equal to the 
        #         fraction_wanted
        else:
            
            # Helps us to find the point (x1, y1) such that the fraction of
            # the area from (x1, y1) to (xs_max, ys_max) is exactly equal to
            # the fraction wanted
            lower_fraction = fraction_at_y_max - fraction_wanted
            
            # Helps us to find the point (x2, y2) such that the fraction of
            # the area from (xs_max, ys_max) to (x2, y2) is exactly equal to
            # the fraction wanted
            upper_fraction = fraction_at_y_max + fraction_wanted
            
            
        # Creates an interpolation function g such that g(xs[i]) gives the
        # fraction of the area under the curve from xs[0] to xs[i] over the
        # total area under the curve. This helps us to find the points
        # x1 and x2 using the three cases above
        interp_x = interpolate.interp1d(cumulative_ys/total_ys, xs, 
                                        fill_value='extrapolate')
        
        # Finds x1 from the lower_fraction and x2 from the upper fraction
        x_vals[:] = interp_x([lower_fraction,upper_fraction])
        
        # Creates an interpolation function to find y1 and y2 from x1 and x2
        interp_y = interpolate.interp1d(xs, ys, fill_value='extrapolate')
        
        # Find y1 and y2
        y_vals[:] = interp_y(x_vals)

        return x_vals, y_vals
    
# Debugging of the function find_area_points
# xs = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
# ys = [-15, -6, 1, 6, 9, 10, 9, 6, 1, -6, -15]
# print(find_area_points(xs, ys, 0.5))
    
    
    
########## FUNCTIONS FOR COORDINATE TRANSFORMATION ##########
"""
This section is ignored as we are working purely in Cartesian coordinates
"""




########## BEAM TRACING FUNCTIONS ##########

def find_normalised_plasma_freq(electron_density, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes the electron plasma frequency \Omega_{pe} and normalizes it
    (dividing it) by the launch angular frequency.
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Separate the calculation into two steps, computing \Omega_{pe} first
      before normalizing to make the code more readable.
    
    INPUT
    ==============================
    electron density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.
    
    OUTPUT
    ==============================
    normalised_plasma_freq (float):
        \Omega_{pe} normalised to the launch frequency.
    """
    
    # Compute \Omega_{pe}
    #
    # Note that the electron density is in units of 10^19 m^{-3}, hence we
    # need to multiply the factor back in
    electron_plasma_freq = constants.e * np.sqrt(
                                electron_density * (10**19)
                                / (constants.epsilon_0 * constants.m_e)
                            )
    
    # Normalizing the plasma frequency by the launch_angular_frequency
    normalised_plasma_freq = electron_plasma_freq/launch_angular_frequency
    
    return normalised_plasma_freq 


def find_normalised_gyro_freq(B_Total, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes the electron gyro frequency (or cyclotron frequency) \Omega_{ce}
    and normalizes it (dividing it) by the launch angular frequency.
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Separate the calculation into two steps, computing \Omega_{ce} first
      before normalizing to make the code more readable.
    
    INPUT
    ==============================
    B_total (float):
        Total background magnetic field.
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma, 
        denoted as \Omega in Valerian's paper.
    
    OUTPUT
    ==============================
    normalised_gyro_freq (float):
        \Omega_{ce} normalised to the launch frequency.
    """
    
    # Compute \Omega_{ce}
    gyro_freq = constants.e * B_Total /constants.m_e
    
    # Normalisation
    normalised_gyro_freq = gyro_freq / launch_angular_frequency  
    
    return normalised_gyro_freq


def find_epsilon_para(electron_density, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes the \epsilon_{bb} component of the cold plasma dielectric tensor
    that is in Valerian's paper. This component is parallel to the magnetic
    field.

    INPUT
    ==============================
    electron density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.

    OUTPUT
    ==============================
    epsilon_para (float):
        the \epsilon_{bb} component of the cold plasma dielectric tensor.
    """
    
    # Computed \Omega_{pe}/\Omega
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, 
                                                         launch_angular_frequency)
    # Compute \epsilon_{bb}
    epsilon_para = 1 - normalised_plasma_freq**2
    
    return epsilon_para


def find_epsilon_perp(electron_density, B_Total, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes the \epsilon_{11} component of the cold plasma dielectric tensor
    that is in Valerian's paper.

    INPUT
    ==============================
    electron density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    B_total (float):
        Total background magnetic field.
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.

    OUTPUT
    ==============================
    epsilon_perp (float):
        the \epsilon_{11} component of the cold plasma dielectric tensor.
    """
    
    # Compute \Omega_{pe}/\Omega
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, 
                                                         launch_angular_frequency)
    
    # Compute \Omega_{ce}/\Omega
    normalised_gyro_freq = find_normalised_gyro_freq(B_Total, 
                                                     launch_angular_frequency)
    
    # Compute \omega_{11}
    epsilon_perp = 1 - normalised_plasma_freq**2 / (1 - normalised_gyro_freq**2)
    
    return epsilon_perp


def find_epsilon_g(electron_density, B_Total, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes the \epsilon_{12} component of the cold plasma dielectric tensor
    that is in Valerian's paper. Note that the cold plasma tensor is symmetric
    therefore \epsilon_{12} = \epsilon_{21}
    
    INPUT
    ==============================
    electron density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    B_total (float):
        Total background magnetic field.
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.
        
    OUTPUT
    ==============================
    epsilon_g (float):
        the \epsilon_{12} component of the cold plasma dielectric tensor.
    """
    
    # Compute \Omega_{pe}/\Omega
    normalised_plasma_freq = find_normalised_plasma_freq(electron_density, 
                                                         launch_angular_frequency)
    
    # Compute \Omega_{ce}/\Omega
    normalised_gyro_freq = find_normalised_gyro_freq(B_Total, 
                                                     launch_angular_frequency)  
    
    # Compute \epsilon_{12}
    epsilon_g = ((normalised_plasma_freq**2) * normalised_gyro_freq 
                 / (1 - normalised_gyro_freq**2)
                 )
    
    return epsilon_g
    

def find_Booker_alpha(electron_density, B_Total, 
                      sin_theta_m_sq, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes alpha of the booker quartic used to compute H_bar. Given by 
    equation (48) of Valerian's paper.

    INPUT
    ==============================
    electron_density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    B_total (float):
        total background magnetic field.
        
    sin_theta_m_sq (float):
        (sin(theta_m))^2 where theta_m is the mismatch angle in Valerian's 
        paper.

    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.

    OUTPUT
    ==============================
    Booker_alpha (float):
        returns the value of alpha in the Booker quartic used to compute the
        value of H_bar.
    """
    
    # Compute epsilon_{bb}
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    
    # Compute epsilon_{11}
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, 
                                     launch_angular_frequency)
    
    # Compute alpha
    # Note that we have wrote the cosine as 1 - sine^2 
    Booker_alpha = (epsilon_para*sin_theta_m_sq 
                    + epsilon_perp*(1-sin_theta_m_sq)
                    )
    
    return Booker_alpha


def find_Booker_beta(electron_density, B_Total, 
                     sin_theta_m_sq, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes beta of the booker quartic used to compute H_bar. Given by 
    equation (49) of Valerian's paper.

    INPUT
    ==============================
    electron_density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    B_total (float):
        total background magnetic field.
        
    sin_theta_m_sq (float):
        (sin(theta_m))^2 where theta_m is the mismatch angle in Valerian's 
        paper.

    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.

    OUTPUT
    ==============================
    Booker_beta (float):
        returns the value of beta in the Booker quartic used to compute the
        value of H_bar.
    """
    
    # Compute epsilon_{bb}
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, 
                                     launch_angular_frequency)
    
    # Compute epsilon_{11}
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    
    # Compute epsilon_{12}
    epsilon_g = find_epsilon_g(electron_density, B_Total, 
                               launch_angular_frequency)
    
    # Compute Booker_beta
    # Note that we have wrote the cosine as 1 - sine^2 
    Booker_beta = (
            - epsilon_perp * epsilon_para * (1+sin_theta_m_sq)
            - (epsilon_perp**2 - epsilon_g**2) * (1-sin_theta_m_sq)
                   )
    
    return Booker_beta


def find_Booker_gamma(electron_density, B_Total, launch_angular_frequency):
    """
    DESCRIPTION
    ==============================
    Computes gamma of the booker quartic used to compute H_bar. Given by 
    equation (50) of Valerian's paper.

    INPUT
    ==============================
    electron_density (float):
        density of the electrons in the plasma, in units of 10^{-19} m^{-3}.
        
    B_total (float):
        total background magnetic field.

    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.

    OUTPUT
    ==============================
    Booker_gamma(float):
        returns the value of gamma in the Booker quartic used to compute the
        value of H_bar.
    """
    
    # Compute epsilon_{bb}
    epsilon_para = find_epsilon_para(electron_density, launch_angular_frequency)
    
    # Compute epsilon_{11}
    epsilon_perp = find_epsilon_perp(electron_density, B_Total, 
                                     launch_angular_frequency)
    
    # Compute epsilon_{12}
    epsilon_g = find_epsilon_g(electron_density, B_Total, 
                               launch_angular_frequency)
    
    # Compute Booker_gamma
    Booker_gamma = epsilon_para*(epsilon_perp**2 - epsilon_g**2)
    
    return Booker_gamma


# FUNCTION DOCUMENTATION NEEDS WORK
#
# Definition of find_density_1D and interp_poloidal_flux still unclear for now
def find_H(q_X, q_Y, q_Z, K_X, K_Y, K_Z, launch_angular_frequency, mode_flag, 
           interp_poloidal_flux, find_density_1D, find_B_X, 
           find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the value of H_bar using equation (47) of Valerian's paper.
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Changed the inputs q_R, q_Z, K_R, K_zeta, K_Z, find_B_R, find_B_T and
    find B_Z to their cartesian counter parts in X, Y, Z.
    
    - Changed the definition of K_magnitude to cartesian coordinates.
    
    - Changed the arguments of interp_poloidal_flux from "q_R, q_Z" to 
      "q_X, q_Y, q_Z".
      
    - Changed the magnetic field functions "find_B_R, find_B_Z" to "find_B_X",
      "find_B_Y" and "find_B_Z". Note that the dependencis of the new magnetic
      field functions, which are cartesian, are "q_X, q_Y, q_Z".
      
    - Changed the definition of B_Total to a cartesian version.
    
    - Changed the definition of b_hat to a cartesian version.
    
    - Changed the definition of K_hat to a cartesian version.
    
    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    K_X (float or vector of shape (n,)):
        X - component of K. Can be a single number or a vector indicating the
        value of K_X at different times.
        
    K_Y (float or vector of shape (n,)):
        Y - component of K. Can be a single number or a vector indicating the
        value of K_Y at different times.
        
    K_Z (float or vector of shape (n,)):
        Z - component of K. Can be a single number or a vector indicating the
        value of K_Z at different times.
        
    launch_angular_frequency (float):
        initial frequency of the microwave beam launched into the plasma,
        denoted as \Omega in Valerian's paper.
        
    mode_flag (+1 or -1):
        This corresponds to plus-or-minus symbol in the quadratic formula. 
        Indicates the X and the O mode, see equation (51) in Valerian's paper.
        
    interp_poloidal_flux (function):
        interpolation function to compute the poloidal flux.
        
    find_density_1D (function):
        computes the electron density based on the poloidal flux.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
    
    
    OUTPUT
    ==============================
    H (float or vector of shape (n,)):
        
    """
    
    # Compute K. Works for K_X, K_Y, K_Z as numbers or (n,) vectors
    K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
    
    # Compute \Omega/c
    wavenumber_K0 = launch_angular_frequency / constants.c
    
    # DON'T UNDERSTAND WHAT THESE DO YET
    poloidal_flux = interp_poloidal_flux(q_X, q_Y, q_Z, grid=False)
    electron_density = find_density_1D(poloidal_flux)
    
    # Calculate the cartesian components of the magnetic field
    # Note that we are using the vector q because we are interested in the 
    # spatial points along the beam, which is defined by q
    #
    # np.squeeze is used to remove any axes of length 1, for example, for a 
    # numpy array of shape (1, 2, 3), the first axes will be removed and the 
    # resultant shape will be (2, 3)
    B_X = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    
    # Compute total magnetic field. Works for B_X, B_Y, B_Z as numbers or 
    # (n,) vectors
    B_Total = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
    
    # Compute unit vector of wavevector. Note that if K_X, K_Y, K_Z are row
    # vectors of shape (n,) then K_magnitude will also be of shape (n,)
    # So now, the array [K_X, K_Y, K_Z] will be of shape (3, n). The below
    # division operation then first divides each element of K_X by the 
    # corresponding element in K_magnitude. The process is then repeated for 
    # K_Y and K_Z. The (3,n) array [K_X_divided, K_Y_divided, K_Z_divided] is 
    # then returned.
    K_hat = np.array([K_X, K_Y, K_Z]) / K_magnitude
    
    # Compute unit vector of magnetic field. Works for B_X, B_Y, B_Z as numbers
    # or (n,) vectors. See explanation for K_hat
    b_hat = np.array([B_X, B_Y, B_Z]) / B_Total
    
    # Compute the square of the sine of the mismatch angle ie. (sin(theta_m))^2
    # based on the definition given in Valerian's paper in Figure 2
    #
    # CASE 1: q_X, q_Y and q_Z are just a single float. This means we are doing
    #         the calculation in a single instance of time
    if np.size(q_X) == 1:
        sin_theta_m_sq = (np.dot(b_hat, K_hat))**2 
        
    # CASE 2: q_X, q_Y and q_Z are vectors of shape (n,), this means
    #         that we have the values of q_X, q_Y and q_Z at multiple instances
    #         of time. Therefore, the calculation is done at all these
    #         instances at once as we would also have K_hat and b_hat of shape
    #         (n,).   
    else: 
        
        # Transpose is needed because we have:
        #   K_hat[0][0] = K_X[0]/ K_magnitude[0]
        #   K_hat[0][1] = K_X[1]/ K_magnitude[0]
        #   K_hat[0][2] = K_X[2]/ K_magnitude[0]
        # But we instead want:
        #   K_hat[0][0] = K_X[0]/ K_magnitude[0]
        #   K_hat[0][1] = K_Y[1]/ K_magnitude[0]
        #   K_hat[0][2] = K_Z[2]/ K_magnitude[0]
        # In order for our contract_special function to work properly
        b_hat = b_hat.T
        K_hat = K_hat.T
        
        # Use the contract_special function on two vectors
        sin_theta_m_sq = (contract_special(b_hat, K_hat))**2 
        
    # Compute Booker alpha, beta, gamma
    Booker_alpha = find_Booker_alpha(electron_density, B_Total, sin_theta_m_sq, 
                                     launch_angular_frequency)
    
    Booker_beta = find_Booker_beta(electron_density, B_Total, sin_theta_m_sq, 
                                   launch_angular_frequency)
    
    Booker_gamma = find_Booker_gamma(electron_density, B_Total,
                                     launch_angular_frequency)
    
    # Compute H (technically H_bar)
    #
    # Note that np.maximum(0, H_discriminant) is used as sometimes
    # H_discriminant ends up being a very small negative number
    H = (K_magnitude/wavenumber_K0)**2 + (
            Booker_beta - mode_flag *
            np.sqrt(np.maximum(
                                np.zeros_like(Booker_beta), 
                                (Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
                            )           
                    )
            ) / (2 * Booker_alpha)

    
    return H




########## INTERFACE FUNCTIONS (VACUUM TO PLASMA) ##########
# Valerian will one day implement from plasma to Vacuum
#
# Note that find_d_poloidal_flux_dX, find_d_poloidal_flux_dY and 
# find_d_poloidal_flux_dZ are the cartesian variant of find_d_poloidal_flux_dR
# and find_d_poloidal_flux_dZ in Valerians code

def find_d_poloidal_flux_dX(q_X, q_Y, q_Z, delta_X, interp_poloidal_flux):
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative d\psi/dX where \psi is the 
    poloidal flux using second order (\delta X^2) forward finite difference 
    (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the x-direction.
        
    interp_poloidal_flux (function):
        interpolating function that computes the poloidal flux at a 
        point in space. (We are interested in points on the central ray of
        the beam, denoted by the position vectors q).

    OUTPUT
    ==============================
    d_poloidal_flux_dX (float or vector of shape (n,)):
        the value of the first order partial derivative in x after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # Compute the zeroth, first and second spatial steps of the poloidal flux
    # in the x direction
    poloidal_flux_0 = interp_poloidal_flux(q_X, q_Y, q_Z)
    poloidal_flux_1 = interp_poloidal_flux(q_X + delta_X, q_Y, q_Z)
    poloidal_flux_2 = interp_poloidal_flux(q_X + 2*delta_X, q_Y, q_Z)
    
    # Apply second order FFD approximation to the first order partial 
    # derivative in x
    d_poloidal_flux_dX = (
                        (-3/2)*poloidal_flux_0 + (2)*poloidal_flux_1 + 
                        (-1/2)*poloidal_flux_2 
                        ) / (delta_X)
    
    return d_poloidal_flux_dX


def find_d_poloidal_flux_dY(q_X, q_Y, q_Z, delta_Y, interp_poloidal_flux):
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative d\psi/dY where \psi is the 
    poloidal flux using second order (\delta Y^2) forward finite difference 
    (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction
        
    interp_poloidal_flux (function):
        interpolating function that computes the poloidal flux at a 
        point in space. (We are interested in points on the central ray of
        the beam, denoted by the position vectors q)

    OUTPUT
    ==============================
    d_poloidal_flux_dY (float or vector of shape (n,)):
        the value of the first order partial derivative in Y after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # Compute the zeroth, first and second spatial steps of the poloidal flux
    # in the x direction
    poloidal_flux_0 = interp_poloidal_flux(q_X, q_Y, q_Z)
    poloidal_flux_1 = interp_poloidal_flux(q_X , q_Y + delta_Y, q_Z)
    poloidal_flux_2 = interp_poloidal_flux(q_X , q_Y + 2*delta_Y, q_Z)
    
    # Apply second order FFD approximation to the first order partial 
    # derivative in x
    d_poloidal_flux_dY = (
                        (-3/2)*poloidal_flux_0 + (2)*poloidal_flux_1 + 
                        (-1/2)*poloidal_flux_2 
                        ) / (delta_Y)
    
    return d_poloidal_flux_dY


def find_d_poloidal_flux_dZ(q_X, q_Y, q_Z, delta_Z, interp_poloidal_flux):

    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative d\psi/dZ where \psi is the 
    poloidal flux using second order (\delta Z^2) forward finite difference 
    (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Z (float):
        spatial step in the z-direction
        
    interp_poloidal_flux (function):
        interpolating function that computes the poloidal flux at a 
        point in space. (We are interested in points on the central ray of
        the beam, denoted by the position vectors q)

    OUTPUT
    ==============================
    d_poloidal_flux_dZ (float or vector of shape (n,)):
        the value of the first order partial derivative after applying second
        order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # Compute the zeroth, first and second spatial steps of the poloidal flux
    # in the z direction
    poloidal_flux_0 = interp_poloidal_flux(q_X, q_Y, q_Z)
    poloidal_flux_1 = interp_poloidal_flux(q_X , q_Y, q_Z + delta_Z)
    poloidal_flux_2 = interp_poloidal_flux(q_X , q_Y, q_Z + 2*delta_Z)
    
    # Apply second order FFD approximation to the first order partial 
    # derivative in z
    d_poloidal_flux_dZ = (
                        (-3/2)*poloidal_flux_0 + (2)*poloidal_flux_1 + 
                        (-1/2)*poloidal_flux_2 
                        ) / (delta_Z)
    
    return d_poloidal_flux_dZ


# ASK VALERIAN
def find_Psi_3D_plasma(Psi_vacuum_3D,
                       zeta, R,
                       K_X, K_Y, 
                       dH_dKX, dH_dKY, dH_dKZ,
                       dH_dX, dH_dY, dH_dZ,
                       d_poloidal_flux_dX, d_poloidal_flux_dY,
                       d_poloidal_flux_dZ):
    """
    DESCRIPTION
    ==============================
    Solves for the components of Psi in the plasma, given the components of 
    Psi in vacuum. The six components, Psi_XX, Psi_XY, Psi_XZ, Psi_YY, Psi_YZ
    and Psi_ZZ are solved from the system of six equations given by (C.10), 
    (C.12), (C.14) and (C.15) in Valerian's paper. However, note that these 
    equations are in CYLINDRICAL coordinates and hence, since we are working
    with cartesian coordinates here, we have utilized equations (66), (67), 
    (68), (69), (70), (71) to convert the six equations into a cartesian 
    variation.
    
    Note that Psi is symmetric, therefore, even though it is a 3x3 matrix,
    only six components (the upper triangular part) needs to be solved.
    
    In the original version of Scotty in cylindrical coordinates, we notice
    that this function did not require specification of the azimuthal angle 
    zeta. The reason is because in tokamak's, where there is cylindrical
    symmetry, the interface is considered as a toroidal surface on the tokamak.
    Therefore, regardless of the toroidal angle zeta, the surface is seen 
    as the same. Of course, the same cannot be said in the general cartesian
    version.

    INPUT
    ==============================
    Psi_vacuum_3D (3x3 numpy array):
        the Psi matrix in vacuum.
        
    zeta (angle in radians):
        azimuthal angle of the interface. Note that if the interface is
        labelled by the coordinates (X, Y, Z), then we have zeta = arctan(Y/X).
        In our case, since we are interested in the beam path, this is likely
        to be q_zeta = arctan(q_Y/q_X)
        
    R (float):
        distance of the interface from origin.  Note that if the interface is
        labelled by the coordinates (X, Y, Z), then we have R=sqrt{x^2 + Y^2}.
        In our case, since we are interested in the beam path, this is likely
        to be q_R
        
    K_X (float):
        X-component of the wave vector.
        
    K_Y (float):
        Y-component of the wave vector.
        
    dH_dKX (float):
        value of the partial derivative of H with respect to KX at the
        plasma-vacuum interface.
        
    dH_dKY (float):
        value of the partial derivative of H with respect to KY at the
        plasma-vacuum interface.
        
    dH_dKZ (float):
        value of the partial derivative of H with respect to KZ at the
        plasma-vacuum interface.
        
    dH_dX (float):
        value of the partial derivative of H with respect to X at the
        plasma-vacuum interface.
        
    dH_dY (float):
        value of the partial derivative of H with respect to Y at the
        plasma-vacuum interface.
        
    dH_dZ (float):
        value of the partial derivative of H with respect to Z at the
        plasma-vacuum interface.
        
    d_poloidal_flux_dX (float):
        value of the partial derivative of poloidal flux psi_P with respect to 
        X at the plasma-vacuum interface.
        
    d_poloidal_flux_dY (float):
        value of the partial derivative of poloidal flux psi_P with respect to 
        Y at the plasma-vacuum interface.
        
    d_poloidal_flux_dZ (float):
        value of the partial derivative of poloidal flux psi_P with respect to 
        Z at the plasma-vacuum interface.
        
    OUTPUT
    ==============================
    Psi_3D_plasma (3x3 numpy array):
        the Psi matrix in plasma.
    """
    
    # Psi components in vacuum
    Psi_v_XX = Psi_vacuum_3D[0,0]
    Psi_v_XY = Psi_vacuum_3D[0,1]
    Psi_v_XZ = Psi_vacuum_3D[0,2]
    Psi_v_YY = Psi_vacuum_3D[1,1]
    Psi_v_YZ = Psi_vacuum_3D[1,2]
    Psi_v_ZZ = Psi_vacuum_3D[2,2]
    
    # Components of the interface matrix. Basically, the system of six
    # equations are written as:
    #   [interface_matrix] * [Psi_p_vector] 
    #   = [vector involving Psi_v in the equations]
    
    interface_matrix = np.zeros([6,6])

    interface_matrix[0, 0] = np.sin(zeta)**2
    interface_matrix[0, 1] = -2 * np.sin(zeta) * np.cos(zeta)
    interface_matrix[0, 3] = np.cos(zeta)**2
    interface_matrix[1, 0] = ((d_poloidal_flux_dZ)**2) * np.cos(zeta)**2
    interface_matrix[1, 1] = (2*((d_poloidal_flux_dZ)**2) * np.sin(zeta) 
                              * np.cos(zeta))
    interface_matrix[1, 2] = (-2 * (d_poloidal_flux_dZ) * (
                                    np.cos(zeta) * d_poloidal_flux_dX + 
                                    np.sin(zeta) * d_poloidal_flux_dY
                                ) * np.cos(zeta)
                            )
    interface_matrix[1, 3] = (d_poloidal_flux_dZ)**2 * (np.sin(zeta))**2
    interface_matrix[1, 4] = (-2 * (d_poloidal_flux_dZ) * (
                                    np.cos(zeta) * d_poloidal_flux_dX + 
                                    np.sin(zeta) * d_poloidal_flux_dY
                                ) * np.sin(zeta)
                            )
    interface_matrix[1, 5] = (
                                np.cos(zeta) * d_poloidal_flux_dX + 
                                np.sin(zeta) * d_poloidal_flux_dY
                            )**2
    interface_matrix[2, 0] = (d_poloidal_flux_dZ * np.sin(zeta) * np.cos(zeta))
    interface_matrix[2, 1] = (-d_poloidal_flux_dZ * (np.cos(zeta)**2 
                                                     - np.sin(zeta)**2)
                              )
    interface_matrix[2, 2] = (-(np.cos(zeta) * d_poloidal_flux_dX + 
                                np.sin(zeta) * d_poloidal_flux_dY)
                                * np.sin(zeta)
                              )
    interface_matrix[2, 3] = (-(d_poloidal_flux_dZ * np.sin(zeta) 
                                * np.cos(zeta))
                              )
    interface_matrix[2, 4] = ((np.cos(zeta) * d_poloidal_flux_dX + 
                                np.sin(zeta) * d_poloidal_flux_dY) 
                              * np.cos(zeta)
                              )
    interface_matrix[3, 0] = ((np.cos(zeta)**2) * (np.cos(zeta) * dH_dKX
                                                   + np.sin(zeta) * dH_dKY)
                              - np.sin(zeta) * np.cos(zeta) *(
                                  - np.sin(zeta) * dH_dKX 
                                  + np.cos(zeta) * dH_dKY 
                                  ) 
                              )
    interface_matrix[3, 1] = ((2 * np.sin(zeta) * np.cos(zeta)) 
                              * (np.cos(zeta) * dH_dKX + np.sin(zeta) * dH_dKY)
                              + (np.cos(zeta)**2 - np.sin(zeta)**2)*(
                                  -np.sin(zeta) * dH_dKX 
                                  + np.cos(zeta) * dH_dKY
                                  )
                              )
    interface_matrix[3, 2] = np.cos(zeta) * dH_dKZ
    interface_matrix[3, 3] = ((np.sin(zeta)**2) * (np.cos(zeta) * dH_dKX 
                                                   + np.sin(zeta) * dH_dKY
                                                   )
                              + (np.sin(zeta) * np.cos(zeta) * (
                                  -np.sin(zeta) * dH_dKX + np.cos(zeta) * dH_dKY
                                  )
                                )
                              )
    interface_matrix[3, 4] = np.sin(zeta) * dH_dKZ 
    interface_matrix[4, 0] = (-np.sin(zeta) * np.cos(zeta) * R * (
                                np.cos(zeta) * dH_dKX + np.sin(zeta) * dH_dKY
                                )
                                +
                                (np.sin(zeta)**2) * (R**2) * (
                                    -(np.sin(zeta)/R) * dH_dKX 
                                    + (np.cos(zeta)/R) * dH_dKY 
                                    )
                            )
    interface_matrix[4, 1] = ((np.cos(zeta)**2 - np.sin(zeta)**2 ) * R *(
                                np.cos(zeta) * dH_dKX + np.sin(zeta) * dH_dKY
                                )
                                - 2 * np.sin(zeta) * np.cos(zeta) * (R**2) * (
                                        -(np.sin(zeta)/R) * dH_dKX 
                                        + (np.cos(zeta)/R) * dH_dKY
                                    )
                            )
    interface_matrix[4, 2] = - np.sin(zeta) * R * dH_dKZ
    interface_matrix[4, 3] = (np.sin(zeta) * np.cos(zeta) * R * (
                                np.cos(zeta) * dH_dKX + np.sin(zeta) * dH_dKY
                                )
                                +
                                (np.cos(zeta)**2) * (R**2) * (
                                    -(np.sin(zeta)/R) * dH_dKX 
                                    + (np.cos(zeta)/R) * dH_dKY 
                                    )
                            )
    
    interface_matrix[4, 4] = np.cos(zeta) * R * dH_dKZ
    
    interface_matrix[5, 2] = (np.cos(zeta) * (
                                np.cos(zeta) * dH_dKX + np.sin(zeta) * dH_dKY
                                ) 
                                - np.sin(zeta) * (
                                    -np.sin(zeta) * dH_dKX 
                                    + np.cos(zeta) * dH_dKY
                                    )
                            )
    interface_matrix[5, 4] = (np.sin(zeta) * (
                                np.cos(zeta) * dH_dKX + np.sin(zeta) * dH_dKY
                                ) 
                                - np.cos(zeta) * (
                                    -np.sin(zeta) * dH_dKX 
                                    + np.cos(zeta) * dH_dKY
                                    )
                            )
    interface_matrix[5, 5] = dH_dKZ
    
    # RHS of the system of equations
    B_vec = np.zeros(6, dtype='complex128')
    B_vec[0] = (Psi_v_XX * np.sin(zeta)**2 
                - 2 * Psi_v_XY * np.sin(zeta) * np.cos(zeta)
                + Psi_v_YY * np.cos(zeta)**2
                )
    B_vec[1] = (
                (d_poloidal_flux_dZ**2) * (np.cos(zeta)**2) * Psi_v_XX 
                + (2 * (d_poloidal_flux_dZ**2) * np.sin(zeta) * np.cos(zeta)
                    * Psi_v_XY)
                + (d_poloidal_flux_dZ**2) * (np.sin(zeta)**2) * Psi_v_YY 
                -(2 * d_poloidal_flux_dZ * (np.cos(zeta) * d_poloidal_flux_dX 
                                            + np.sin(zeta) * d_poloidal_flux_dY
                                            ) * np.cos(zeta) * Psi_v_XZ 
                  ) 
                - (2 * d_poloidal_flux_dZ * (np.cos(zeta) * d_poloidal_flux_dX 
                                            + np.sin(zeta) * d_poloidal_flux_dY
                                                ) * np.sin(zeta) * Psi_v_YZ
                   )
                + ((np.cos(zeta) * d_poloidal_flux_dX + 
                   np.sin(zeta) * d_poloidal_flux_dY)**2) * Psi_v_ZZ
                )
    B_vec[2] = (d_poloidal_flux_dZ * np.sin(zeta) * np.cos(zeta) * Psi_v_XX
                - (d_poloidal_flux_dZ 
                   * (np.cos(zeta)**2 - np.sin(zeta)**2) * Psi_v_XY
                   )
                - d_poloidal_flux_dZ * np.sin(zeta) * np.cos(zeta) * Psi_v_YY
                - (np.cos(zeta) * d_poloidal_flux_dX 
                   + np.sin(zeta) * d_poloidal_flux_dY
                   ) * np.cos(zeta) * Psi_v_XZ
                + (np.cos(zeta) * d_poloidal_flux_dX 
                   + np.sin(zeta) * d_poloidal_flux_dY
                   ) * np.cos(zeta) * Psi_v_YZ
                )
    B_vec[3] = (np.cos(zeta) * dH_dX + np.sin(zeta) * dH_dY + (
                    K_X * np.sin(zeta) - K_Y * np.cos(zeta)
                    ) * (
                        -(np.sin(zeta)/R) * dH_dKX + (np.cos(zeta)) * dH_dKY
                        )
                )
    B_vec[4] = (-(K_X * np.cos(zeta) + K_Y * np.sin(zeta)) 
                * (-np.sin(zeta) * dH_dKX + np.cos(zeta) * dH_dKY)
                )
    B_vec[5] = - dH_dZ 
    
    # Solve the system of equations for Psi in plasma
    #
    # Note that the interface_matrix will be singular if one tries to transition while still in vacuum (and there's no plasma at all)
    # at least that's what happens, in my experience
    interface_matrix_inverse = np.linalg.inv(interface_matrix)
    
    [Psi_p_XX, Psi_p_XY, Psi_p_XZ, 
     Psi_p_YY, Psi_p_YZ, Psi_p_ZZ] = np.matmul(interface_matrix_inverse, B_vec)
    
    # Create the Psi matrix in the plasma (remember that it is symmetric)
    Psi_3D_plasma = np.zeros([3,3],dtype='complex128')
    Psi_3D_plasma[0,0] = Psi_p_XX
    Psi_3D_plasma[1,1] = Psi_p_YY
    Psi_3D_plasma[2,2] = Psi_p_ZZ
    Psi_3D_plasma[0,1] = Psi_p_XY
    Psi_3D_plasma[1,0] = Psi_3D_plasma[0,1]
    Psi_3D_plasma[0,2] = Psi_p_XZ
    Psi_3D_plasma[2,0] = Psi_3D_plasma[0,2]
    Psi_3D_plasma[1,2] = Psi_p_YZ
    Psi_3D_plasma[2,1] = Psi_3D_plasma[1,2]
    
    return Psi_3D_plasma 


########## FUNCTIONS FOR ANALYSIS ##########
# 
# Note that find_dbhat_dX, find_dbhat_dY and find_dbhat_dZ are the cartesian
# variants of find_dbhat_dR and find_dbhat_dZ in Valerians code


def find_dbhat_dX(q_X, q_Y, q_Z, delta_X, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Compute the first order partial derivative derivative of b_hat (unit vector 
    of the magnetic field) in the x-direction using second order (delta_X^2)  
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the x-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dbhat_dX (float or vector of shape (n,)):
        the value of the first order partial derivative in x after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_X = np.squeeze(find_B_X(q_X + delta_X, q_Y, q_Z))
    B_Y_plus_X = np.squeeze(find_B_Y(q_X + delta_X, q_Y, q_Z))
    B_Z_plus_X = np.squeeze(find_B_Z(q_X + delta_X, q_Y, q_Z))
    
    B_magnitude_plus = np.sqrt(B_X_plus_X**2 + B_Y_plus_X**2 + B_Z_plus_X**2)
    b_hat_plus = (
                    np.array([B_X_plus_X, B_Y_plus_X, B_Z_plus_X]) 
                    / B_magnitude_plus
                    )
    
    # n-1 step in the CFD
    B_X_minus_X = np.squeeze(find_B_X(q_X - delta_X, q_Y, q_Z))
    B_Y_minus_X = np.squeeze(find_B_Y(q_X - delta_X, q_Y, q_Z))
    B_Z_minus_X = np.squeeze(find_B_Z(q_X - delta_X, q_Y, q_Z))
    
    B_magnitude_minus = np.sqrt(B_X_minus_X**2 + B_Y_minus_X**2 + B_Z_minus_X**2)
    b_hat_minus = (
                    np.array([B_X_minus_X, B_Y_minus_X, B_Z_minus_X]) 
                    / B_magnitude_minus
                    )
    
    # Apply CFD approximation
    #
    # Note that if q_X, q_Y, q_Z are vectors of shape (n,), then so will
    # dbhat_dX, which is simply the value of dbhat_dX at different times.
    dbhat_dX = (b_hat_plus - b_hat_minus) / (2 * delta_X)
           
    return dbhat_dX



def find_dbhat_dY(q_X, q_Y, q_Z, delta_Y, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Compute the first order partial derivative derivative of b_hat (unit vector 
    of the magnetic field) in the y-direction using second order (delta_Y^2)
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dbhat_dY (float or vector of shape (n,)):
        the value of the first order partial derivative in y after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_Y = np.squeeze(find_B_X(q_X, q_Y + delta_Y, q_Z))
    B_Y_plus_Y = np.squeeze(find_B_Y(q_X, q_Y + delta_Y, q_Z))
    B_Z_plus_Y = np.squeeze(find_B_Z(q_X, q_Y + delta_Y, q_Z))
    
    B_magnitude_plus = np.sqrt(B_X_plus_Y**2 + B_Y_plus_Y**2 + B_Z_plus_Y**2)
    b_hat_plus = (
                    np.array([B_X_plus_Y, B_Y_plus_Y, B_Z_plus_Y]) 
                    / B_magnitude_plus
                    )
    
    # n-1 step in the CFD
    B_X_minus_Y = np.squeeze(find_B_X(q_X, q_Y - delta_Y, q_Z))
    B_Y_minus_Y = np.squeeze(find_B_Y(q_X, q_Y - delta_Y, q_Z))
    B_Z_minus_Y = np.squeeze(find_B_Z(q_X, q_Y - delta_Y, q_Z))
    
    B_magnitude_minus = np.sqrt(B_X_minus_Y**2 + B_Y_minus_Y**2 + B_Z_minus_Y**2)
    b_hat_minus = (
                    np.array([B_X_minus_Y, B_Y_minus_Y, B_Z_minus_Y]) 
                    / B_magnitude_minus
                    )
    
    # Apply CFD approximation
    #
    # Note that if q_X, q_Y, q_Z are vectors of shape (n,), then so will
    # dbhat_dY, which is simply the value of dbhat_dY at different times.
    dbhat_dY = (b_hat_plus - b_hat_minus) / (2 * delta_Y)
           
    return dbhat_dY


def find_dbhat_dZ(q_X, q_Y, q_Z, delta_Z, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Compute the first order partial derivative derivative of b_hat (unit vector 
    of the magnetic field) in the z-direction using second order (delta_Z^2) 
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Z (float):
        spatial step in the z-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dbhat_dZ (float or vector of shape (n,)):
        the value of the first order partial derivative in z after applying 
        second order CFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_Z = np.squeeze(find_B_X(q_X, q_Y, q_Z + delta_Z))
    B_Y_plus_Z = np.squeeze(find_B_Y(q_X, q_Y, q_Z + delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z + delta_Z))
    
    B_magnitude_plus = np.sqrt(B_X_plus_Z**2 + B_Y_plus_Z**2 + B_Z_plus_Z**2)
    b_hat_plus = (
                    np.array([B_X_plus_Z, B_Y_plus_Z, B_Z_plus_Z]) 
                    / B_magnitude_plus
                    )
    
    # n-1 step in the CFD
    B_X_minus_Z = np.squeeze(find_B_X(q_X, q_Y, q_Z - delta_Z))
    B_Y_minus_Z = np.squeeze(find_B_Y(q_X, q_Y, q_Z - delta_Z))
    B_Z_minus_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z - delta_Z))
    
    B_magnitude_minus = np.sqrt(B_X_minus_Z**2 + B_Y_minus_Z**2 + B_Z_minus_Z**2)
    b_hat_minus = (
                    np.array([B_X_minus_Z, B_Y_minus_Z, B_Z_minus_Z]) 
                    / B_magnitude_minus
                    )
    
    # Apply CFD approximation
    #
    # Note that if q_X, q_Y, q_Z are vectors of shape (n,), then so will
    # dbhat_dZ, which is simply the value of dbhat_dZ at different times.
    dbhat_dZ = (b_hat_plus - b_hat_minus) / (2 * delta_Z)
           
    return dbhat_dZ


def find_D(K_magnitude, launch_angular_frequency, epsilon_para, epsilon_perp,
           epsilon_g, theta_m):
    """
    DESCRIPTION
    ==============================
    Computes the components of the D matrix based on equation (36) in 
    Valerian's paper.

    INPUT
    ==============================
    K_magnitude (float):
        magnitude of the probe beam wave vector.
        
    launch_angular_frequency (float):
        initial angular frequency when the probe beam is first launched.
        
    epsilon_para (float):
        epsilon_{bb} in Valerian's paper.
        
    epsilon_perp (float):
        epsilon_{11} in Valerian's paper.
    
    epsilon_g (float):
        epsilon_{12} in Valerian's paper.
        
    theta_m (float):
        mismatch angle.

    OUTPUT
    ==============================
    
    Relevant components of the D matrix (list):
            Required components of the D matrix, namely, D_11, D_22, D_bb
            D_12, D_1b.

    """
    
    # Compute K_0 = \Omega/c
    wavenumber_K0 = launch_angular_frequency / constants.c
    
    # Compute K/K_0
    n_ref_index = K_magnitude/wavenumber_K0
    
    # Compute sin(theta_m) and cosine(theta_m)
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)
    
    # Compute the components using equation (37), (38), (39), (40), (41) of
    # Valerian's paper
    D_11_component = epsilon_perp - n_ref_index**2*sin_theta_m**2
    D_22_component = epsilon_perp - n_ref_index**2
    D_bb_component = epsilon_para - n_ref_index**2*cos_theta_m**2
    D_12_component = epsilon_g
    D_1b_component = n_ref_index**2*sin_theta_m*cos_theta_m
    
    return (
            [D_11_component, D_22_component, D_bb_component, 
            D_12_component, D_1b_component]
            )


def find_H_Cardano(K_magnitude, launch_angular_frequency, epsilon_para, 
               epsilon_perp, epsilon_g, theta_m):
    """
    DESCRIPTION
    ==============================
    Finds the three possible solutions of H based on appendix B of Valerian's
    paper. This function is meant to be used in post-processing
    
    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Added the "D_components" variable to make the code more readable

    INPUT
    ==============================
    K_magnitude (float):
        magnitude of the probe beam wave vector.
        
    launch_angular_frequency (float):
        initial angular frequency when the probe beam is first launched.
        
    epsilon_para (float):
        epsilon_{bb} in Valerian's paper.
        
    epsilon_perp (float):
        epsilon_{11} in Valerian's paper.
    
    epsilon_g (float):
        epsilon_{12} in Valerian's paper.
        
    theta_m (float):
        mismatch angle.

    OUTPUT
    ==============================
    H_1_Cardano, H_2_Cardano, H_3_Cardano (float):
        the 3 solutions of H as described in appendix B of Valerians paper
    """
    
    # Compute components of the D-matrix
    D_components = find_D(K_magnitude, launch_angular_frequency, epsilon_para,
                          epsilon_perp, epsilon_g, theta_m)
    
    D_11_component = D_components[0]
    D_22_component = D_components[1]
    D_bb_component = D_components[2]
    D_12_component = D_components[3]
    D_1b_component = D_components[4]
    
    # Compute the h2, h1 and h0 coefficients given by equations (B.3), (B.4)
    # and (B.5) respectively in Valerian's paper
    h_2_coefficient = - D_11_component - D_22_component - D_bb_component
    
    h_1_coefficient = (
                        D_11_component * D_bb_component 
                       + D_11_component * D_22_component 
                       + D_22_component * D_bb_component 
                       - D_12_component**2 - D_1b_component**2
                       )
    
    h_0_coefficient = (D_22_component * (D_1b_component**2)
                       + D_bb_component * (D_12_component**2)
                       - D_11_component * D_22_component * D_bb_component)
    
    # Compute the ht coefficient given by equation (B.9) in Valerian's paper
    #
    # Note that we have added a "+0j" inside the np.sqrt() to make the argument 
    # of the np.sqrt complex, so that the sqrt evaluates negative functions
    h_t_coefficient = (
                            -2*h_2_coefficient**3
                            +9*h_2_coefficient*h_1_coefficient
                            -27*h_0_coefficient
                            +3*np.sqrt(3)*np.sqrt(
                                4*h_2_coefficient**3 * h_0_coefficient
                                -h_2_coefficient**2 * h_1_coefficient**2
                                -18*h_2_coefficient * h_1_coefficient 
                                * h_0_coefficient
                                +4*h_1_coefficient**3
                                +27*h_0_coefficient**2
                                +0j 
                            )
                        )**(1/3)
    
    # Compute H_1_Cardano, H_2_Cardano and H_3_Cardano
    H_1_Cardano = (
                    h_t_coefficient/(3*2**(1/3)) 
                   - (
                       2**(1/3) *(3*h_1_coefficient - h_2_coefficient**2)
                      /(3*h_t_coefficient)
                      ) 
                   - h_2_coefficient/3)
    
    H_2_Cardano = (- (1 - 1j*np.sqrt(3))/(6*2**(1/3))*h_t_coefficient 
                   + (
                       (1 + 1j*np.sqrt(3))*(3*h_1_coefficient 
                      - h_2_coefficient**2)/(3*2**(2/3)*h_t_coefficient)
                      ) 
                   - h_2_coefficient/3)
    
    H_3_Cardano = (- (1 + 1j*np.sqrt(3))/(6*2**(1/3))*h_t_coefficient 
                   + (
                       (1 - 1j*np.sqrt(3))*(3*h_1_coefficient 
                     - h_2_coefficient**2)/(3*2**(2/3)*h_t_coefficient)
                        ) 
                   - h_2_coefficient/3)
    
    return H_1_Cardano, H_2_Cardano, H_3_Cardano




########## FUNCTIONS FOR CIRCULAR GAUSSIAN BEAM IN VACUUM ##########

def find_waist(width, wavenumber, curvature): 
    """
    DESCRIPTION
    ==============================
    Computes the minimal width (waist) of the circular Gaussian beam in vacuum.
    Note that the minimal width occurs at the focal point of the lens used
    to focus the Gaussian beam. The intensity here is the greatest.
    
    Note that this solution is directly taken from:
    https://physics.stackexchange.com/questions/270249/for-a-gaussian-beam-how-can-i-calculate-w-0-z-and-z-r-from-knowledge-of

    INPUT
    ==============================
    width (float):
        this is the current known width, somewhere away from the location
        of the beam waist.
        
    wavenumber (float):
        wavenumber of the Gaussian beam.
        
    curvature (float):
        this is 1/R where R is the radius of curvature of the phase fronts.

    OUTPUT
    ==============================
    waist (float):
        length of the waist of the Gaussian beam.
    """
    waist = width / np.sqrt(1+curvature**2 * width**4 * wavenumber**2 / 4)
    return waist


def find_Rayleigh_length(waist, wavenumber): 
    """
    DESCRIPTION
    ==============================
    Computes the Rayleigh length of a circular Gaussian beam in vacuum. This is
    the point where the Gaussian beam width, w, increases by a factor of 
    \sqrt{2} from the beam waist w_0 (minimal beam width).

    INPUT
    ==============================
    waist (float):
        the minimal width of the circular Gaussian beam in vacuum. Achieved
        at the focus point of the lens.
        
    wavenumber (float):
        wavenumber of the Gaussian beam.

    OUTPUT
    ==============================
    Rayleigh_length (float):
        the Rayleigh_length
    """
    
    Rayleigh_length = 0.5 * wavenumber * waist**2
    return Rayleigh_length


def find_distance_from_waist(width, wavenumber, curvature):
    """
    DESCRIPTION
    ==============================
    Based on the current width of the beam for some location away from the
    waist, compute the distance from the waist. Basically, solves for z in 
    the formula:
        w = w0 \sqrt{1 + (z/z_R)^2}
    where z_R is the Rayleigh length, w is the beam width at z and w0 is the
    length of the beam waist.
    
    MODIFICATIONS FROM VALERIAN'S ORIGINAL CODE
    ==============================
    - Rather then using the raw formula for w0 and z_R, I use the previously
      defined function find_waist and find_Rayleigh_length
    
    INPUT
    ==============================
     width (float):
         this is the current known width, somewhere away from the location
         of the beam waist.
         
     wavenumber (float):
         wavenumber of the Gaussian beam.
         
     curvature (float):
         this is 1/R where R is the radius of curvature of the phase fronts.

    OUTPUT
    ==============================
    distance_from_waist (float):
        axial distance from the waist of the Gaussian beam.

    """
    
    # compute waist 
    w0 = find_waist(width, wavenumber, curvature)
    
    # compute Rayleigh length
    z_R = find_Rayleigh_length(w0, wavenumber)
    
    # compute axial distance
    distance_from_waist = np.sign(curvature)*np.sqrt(
                                                     z_R * ((width/w0)**2 - 1)
                                                     )
    return distance_from_waist


# CHECK WITH VALERIAN
def propagate_circular_beam(distance, wavenumber, w0):
    """
    DESCRIPTION
    ==============================
    Based on the current axial distance, z,  computes the radius of curvature
    and beam widths using the following formulas:
        w = w0 \sqrt{1 + (z/z_R)^2}
        curvature = 1/R
    where R is the radius of curvature given by:
        R = (1/z)*(z^2 + z_R^2)
        
    Note that if the inputs are vectors, then we are attempting to compute
    w and R for multiple points along the axial direction.

    MODIFICATIONS FROM VALERIANS ORIGINAL CODE
    ==============================
    - Introduced the line "z = distance" to relate better to the equations

    INPUT
    ==============================
    - distance (float or vector of shape (1, n)):
        axial distance from the waist.
        
    - wavenumber (float):
        wavenumber of the Gaussian beam.
        
    - w0 (float):
        size of the waist.

    OUTPUT
    ==============================
    widths (float or vector of shape (1, n)):
        widths corresponding to the given axial direction.
        
    curvatures (float or vector of shape (1, n)):
        curvatures corresponding to the given axial direction.
    """
    
    # Compute the Rayleigh length
    z_R = find_Rayleigh_length(w0, wavenumber)
    
    # For code readability, label the axial distance at z
    z = distance
    
    # Computes the widths at each axial distance
    widths = w0 * np.sqrt(1+(z/z_R)**2)
    
    # Computes the curvatures
    curvatures = z/(z**2 + z_R**2)
    
    return widths, curvatures




########## FUNCTIONS FOR GENERAL GAUSSIAN BEAM IN VACUUM ##########
"""
This part is to be filled in after discussing with valerian about the circular
Gaussian beam
"""




########## FUNCTIONS FOR DEBUGGING ##########
#
# As compared to Valerian's original code, the codes below have been modified
# to suit cartesian coordinates

### First Order Positional Derivatives of the Magnetic Field (CFD) ###

def find_dB_dX_CFD(q_X, q_Y, q_Z, delta_X, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative of the magnitude of the 
    magnetic field, B, with respect to x using second order (delta_X^2) 
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the x-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dB_dX (float or or vector of shape (n,)):
        the value of the first order partial derivative of the magnetic field
        with respect to x using second order CFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_X = np.squeeze(find_B_X(q_X + delta_X, q_Y, q_Z))
    B_Y_plus_X = np.squeeze(find_B_Y(q_X + delta_X, q_Y, q_Z))
    B_Z_plus_X = np.squeeze(find_B_Z(q_X + delta_X, q_Y, q_Z))
    
    B_magnitude_plus = np.sqrt(B_X_plus_X**2 + B_Y_plus_X**2 + B_Z_plus_X**2)
    
    # n-1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_minus_X = np.squeeze(find_B_X(q_X - delta_X, q_Y, q_Z))
    B_Y_minus_X = np.squeeze(find_B_Y(q_X - delta_X, q_Y, q_Z))
    B_Z_minus_X = np.squeeze(find_B_Z(q_X - delta_X, q_Y, q_Z))
    
    B_magnitude_minus = np.sqrt(
                                B_X_minus_X**2 + B_Y_minus_X**2 
                                + B_Z_minus_X**2
                                )
    
    # Compute the value of the partial derivative using second order CFD
    dB_dX = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_X)  
    
    return dB_dX


def find_dB_dY_CFD(q_X, q_Y, q_Z, delta_Y, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative of the magnitude of the 
    magnetic field, B, with respect to y using second order (delta_Y^2) 
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dB_dX (float or or vector of shape (n,)):
        the value of the first order partial derivative of the magnetic field
        with respect to y using second order CFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_Y = np.squeeze(find_B_X(q_X, q_Y + delta_Y, q_Z))
    B_Y_plus_Y = np.squeeze(find_B_Y(q_X, q_Y + delta_Y, q_Z))
    B_Z_plus_Y = np.squeeze(find_B_Z(q_X, q_Y + delta_Y, q_Z))
    
    B_magnitude_plus = np.sqrt(B_X_plus_Y**2 + B_Y_plus_Y**2 + B_Z_plus_Y**2)
    
    # n-1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_minus_Y = np.squeeze(find_B_X(q_X, q_Y - delta_Y, q_Z))
    B_Y_minus_Y = np.squeeze(find_B_Y(q_X, q_Y - delta_Y, q_Z))
    B_Z_minus_Y = np.squeeze(find_B_Z(q_X, q_Y - delta_Y, q_Z))
    
    B_magnitude_minus = np.sqrt(
                                B_X_minus_Y**2 + B_Y_minus_Y**2 
                                + B_Z_minus_Y**2
                                )
    
    # Compute the value of the partial derivative using second order CFD
    dB_dY = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_Y)  
    
    return dB_dY


def find_dB_dZ_CFD(q_X, q_Y, q_Z, delta_Z, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative of the magnitude of the 
    magnetic field, B, with respect to z using second order (delta_Z^2) 
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dB_dX (float or or vector of shape (n,)):
        the value of the first order partial derivative of the magnetic field
        with respect to z using second order CFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_Z = np.squeeze(find_B_X(q_X, q_Y, q_Z + delta_Z))
    B_Y_plus_Z = np.squeeze(find_B_Y(q_X, q_Y, q_Z + delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z + delta_Z))
    
    B_magnitude_plus = np.sqrt(B_X_plus_Z**2 + B_Y_plus_Z**2 + B_Z_plus_Z**2)
    
    # n-1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_minus_Z = np.squeeze(find_B_X(q_X, q_Y, q_Z - delta_Z))
    B_Y_minus_Z = np.squeeze(find_B_Y(q_X, q_Y, q_Z - delta_Z))
    B_Z_minus_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z - delta_Z))
    
    B_magnitude_minus = np.sqrt(
                                B_X_minus_Z**2 + B_Y_minus_Z**2 
                                + B_Z_minus_Z**2
                                )
    
    # Compute the value of the partial derivative using second order CFD
    dB_dZ = (B_magnitude_plus - B_magnitude_minus) / (2 * delta_Z)  
    
    return dB_dZ


### First Order Positonal Derivatives of the Magnetic Field (FFD) ###

def find_dB_dX_FFD(q_X, q_Y, q_Z, delta_X, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative of the magnitude of the 
    magnetic field, B, with respect to x using second order (delta_X^2) 
    forward finite difference (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the x-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dB_dX (float or or vector of shape (n,)):
        the value of the first order partial derivative of the magnetic field
        with respect to x using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_1 = np.squeeze(find_B_X(q_X + delta_X, q_Y, q_Z))
    B_Y_1 = np.squeeze(find_B_Y(q_X + delta_X, q_Y, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_X + delta_X, q_Y, q_Z))
    
    B_magnitude_1 = np.sqrt(B_X_1**2 + B_Y_1**2 + B_Z_1**2)
    
    # n+2 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_2 = np.squeeze(find_B_X(q_X + 2*delta_X, q_Y, q_Z))
    B_Y_2 = np.squeeze(find_B_Y(q_X + 2*delta_X, q_Y, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_X + 2*delta_X, q_Y, q_Z))
    
    B_magnitude_2 = np.sqrt(B_X_2**2 + B_Y_2**2 + B_Z_2**2)
    
    # Compute the value of the partial derivative using second order FFD
    dB_dX = (
                ( (-3/2)*B_magnitude_0 + (2)*B_magnitude_1 
                 + (-1/2)*B_magnitude_2 ) / (delta_X)
            )
    
    return dB_dX


def find_dB_dY_FFD(q_X, q_Y, q_Z, delta_Y, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative of the magnitude of the 
    magnetic field, B, with respect to y using second order (delta_Y^2) 
    forward finite difference (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dB_dX (float or or vector of shape (n,)):
        the value of the first order partial derivative of the magnetic field
        with respect to y using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_1 = np.squeeze(find_B_X(q_X, q_Y + delta_Y, q_Z))
    B_Y_1 = np.squeeze(find_B_Y(q_X, q_Y + delta_Y, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_X, q_Y + delta_Y, q_Z))
    
    B_magnitude_1 = np.sqrt(B_X_1**2 + B_Y_1**2 + B_Z_1**2)
    
    # n+2 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_2 = np.squeeze(find_B_X(q_X, q_Y + 2*delta_Y, q_Z))
    B_Y_2 = np.squeeze(find_B_Y(q_X, q_Y + 2*delta_Y, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_X, q_Y + 2*delta_Y, q_Z))
    
    B_magnitude_2 = np.sqrt(B_X_2**2 + B_Y_2**2 + B_Z_2**2)
    
    # Compute the value of the partial derivative using second order FFD
    dB_dY = (
                ( (-3/2)*B_magnitude_0 + (2)*B_magnitude_1 
                 + (-1/2)*B_magnitude_2 ) / (delta_Y)
            )
    
    return dB_dY


def find_dB_dZ_FFD(q_X, q_Y, q_Z, delta_Z, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the first order partial derivative of the magnitude of the 
    magnetic field, B, with respect to z using second order (delta_Z^2) 
    forward finite difference (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Z (float):
        spatial step in the z-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    dB_dX (float or or vector of shape (n,)):
        the value of the first order partial derivative of the magnetic field
        with respect to z using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_1 = np.squeeze(find_B_X(q_X, q_Y, q_Z + delta_Z))
    B_Y_1 = np.squeeze(find_B_Y(q_X, q_Y, q_Z + delta_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_X, q_Y, q_Z + delta_Z))
    
    B_magnitude_1 = np.sqrt(B_X_1**2 + B_Y_1**2 + B_Z_1**2)
    
    # n+2 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_2 = np.squeeze(find_B_X(q_X, q_Y, q_Z + 2*delta_Z))
    B_Y_2 = np.squeeze(find_B_Y(q_X, q_Y, q_Z + 2*delta_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_X, q_Y, q_Z + 2*delta_Z))
    
    B_magnitude_2 = np.sqrt(B_X_2**2 + B_Y_2**2 + B_Z_2**2)
    
    # Compute the value of the partial derivative using second order FFD
    dB_dZ = (
                ( (-3/2)*B_magnitude_0 + (2)*B_magnitude_1 
                 + (-1/2)*B_magnitude_2 ) / (delta_Z)
            )
    
    return dB_dZ


### Second Order Positional Derivatives of the Magnetic Field (CFD) ###

def find_d2B_dX2_CFD(q_X, q_Y, q_Z, delta_X, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the magnitude of the 
    magnetic field, B, with respect to x using second order (delta_X^2)
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dX2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the magnetic field
        with respect to x using second order CFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_X = np.squeeze(find_B_X(q_X + delta_X, q_Y, q_Z))
    B_Y_plus_X = np.squeeze(find_B_Y(q_X + delta_X, q_Y, q_Z))
    B_Z_plus_X = np.squeeze(find_B_Z(q_X + delta_X, q_Y, q_Z))

    B_magnitude_plus = np.sqrt(B_X_plus_X**2 + B_Y_plus_X**2 + B_Z_plus_X**2)
    
    # n-1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_minus_X = np.squeeze(find_B_X(q_X - delta_X, q_Y, q_Z))
    B_Y_minus_X = np.squeeze(find_B_Y(q_X - delta_X, q_Y, q_Z))
    B_Z_minus_X = np.squeeze(find_B_Z(q_X - delta_X, q_Y, q_Z))

    B_magnitude_minus = np.sqrt(B_X_minus_X**2 + B_Y_minus_X**2 
                                + B_Z_minus_X**2)
  
    
    # Compute the second order partial derivative 
    d2B_dX2 = (
                ( 
                (1)*B_magnitude_minus 
                + (-2)*B_magnitude_0 
                + (1)*B_magnitude_plus 
                 ) / (delta_X**2)
            )
    
    return d2B_dX2


def find_d2B_dY2_CFD(q_X, q_Y, q_Z, delta_Y, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the magnitude of the 
    magnetic field, B, with respect to y using second order (detla_Y^2) 
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dY2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the magnetic field
        with respect to y using second order CFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_plus_Y = np.squeeze(find_B_X(q_X, q_Y + delta_Y, q_Z))
    B_Y_plus_Y = np.squeeze(find_B_Y(q_X, q_Y + delta_Y, q_Z))
    B_Z_plus_Y = np.squeeze(find_B_Z(q_X, q_Y + delta_Y, q_Z))

    B_magnitude_plus = np.sqrt(B_X_plus_Y**2 + B_Y_plus_Y**2 + B_Z_plus_Y**2)
    
    # n-1 step in the CFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_minus_Y = np.squeeze(find_B_X(q_X, q_Y - delta_Y, q_Z))
    B_Y_minus_Y = np.squeeze(find_B_Y(q_X, q_Y - delta_Y, q_Z))
    B_Z_minus_Y = np.squeeze(find_B_Z(q_X, q_Y - delta_Y, q_Z))

    B_magnitude_minus = np.sqrt(B_X_minus_Y**2 + B_Y_minus_Y**2 
                                + B_Z_minus_Y**2)
  
    
    # Compute the second order partial derivative 
    d2B_dY2 = (
                ( 
                (1)*B_magnitude_minus 
                + (-2)*B_magnitude_0 
                + (1)*B_magnitude_plus 
                 ) / (delta_Y**2)
            )
    
    return d2B_dY2


def find_d2B_dZ2_CFD(q_X, q_Y, q_Z, delta_Z, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the magnitude of the 
    magnetic field, B, with respect to z using second order (delta_Z^2)
    central finite difference (CFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Z (float):
        spatial step in the z-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dZ2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the magnetic field
        with respect to z using second order CFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step in the CFD:
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the CFD
    B_X_plus_Z = np.squeeze(find_B_X(q_X, q_Y, q_Z + delta_Z))
    B_Y_plus_Z = np.squeeze(find_B_Y(q_X, q_Y, q_Z + delta_Z))
    B_Z_plus_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z + delta_Z))

    B_magnitude_plus = np.sqrt(B_X_plus_Z**2 + B_Y_plus_Z**2 + B_Z_plus_Z**2)
    
    # n-1 step in the CFD
    
    B_X_minus_Z = np.squeeze(find_B_X(q_X, q_Y, q_Z - delta_Z))
    B_Y_minus_Z = np.squeeze(find_B_Y(q_X, q_Y, q_Z - delta_Z))
    B_Z_minus_Z = np.squeeze(find_B_Z(q_X, q_Y, q_Z - delta_Z))

    B_magnitude_minus = np.sqrt(B_X_minus_Z**2 + B_Y_minus_Z**2 
                                + B_Z_minus_Z**2)
  
    
    # Compute the second order partial derivative 
    d2B_dZ2 = (
                ( 
                (1)*B_magnitude_minus 
                + (-2)*B_magnitude_0 
                + (1)*B_magnitude_plus 
                 ) / (delta_Z**2)
            )
    
    return d2B_dZ2


### Second Order Positional Derivatives of the Magnetic Field (FFD) ###


def find_d2B_dX2_FFD(q_X, q_Y, q_Z, delta_X, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the magnitude of the 
    magnetic field, B, with respect to x using second order (delta_X^2) 
    forward finite difference (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the x-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dX2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the magnetic field
        with respect to x using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_1 = np.squeeze(find_B_X(q_X + delta_X, q_Y, q_Z))
    B_Y_1 = np.squeeze(find_B_Y(q_X + delta_X, q_Y, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_X + delta_X, q_Y, q_Z))
    
    B_magnitude_1 = np.sqrt(B_X_1**2 + B_Y_1**2 + B_Z_1**2)
    
    # n+2 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_2 = np.squeeze(find_B_X(q_X + 2*delta_X, q_Y, q_Z))
    B_Y_2 = np.squeeze(find_B_Y(q_X + 2*delta_X, q_Y, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_X + 2*delta_X, q_Y, q_Z))
    
    B_magnitude_2 = np.sqrt(B_X_2**2 + B_Y_2**2 + B_Z_2**2)
    
    # n+3 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_3 = np.squeeze(find_B_X(q_X + 3*delta_X, q_Y, q_Z))
    B_Y_3 = np.squeeze(find_B_Y(q_X + 3*delta_X, q_Y, q_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_X + 3*delta_X, q_Y, q_Z))
    
    B_magnitude_3 = np.sqrt(B_X_3**2 + B_Y_3**2 + B_Z_3**2)
    
    
    # Compute the value of the partial derivative using second order FFD
    d2B_dX2 = (
                ( 
                    (2)*B_magnitude_0 + (-5)*B_magnitude_1 
                    + (4)*B_magnitude_2 + (-1)*B_magnitude_3 
                ) / (delta_X**2)
            )
    
    return d2B_dX2


def find_d2B_dY2_FFD(q_X, q_Y, q_Z, delta_Y, find_B_X, find_B_Y, find_B_Z): 
    
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the magnitude of the 
    magnetic field, B, with respect to y using second order (delta_Y^2) 
    forward finite difference (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dY2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the magnetic field
        with respect to y using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_1 = np.squeeze(find_B_X(q_X, q_Y + delta_Y, q_Z))
    B_Y_1 = np.squeeze(find_B_Y(q_X, q_Y + delta_Y, q_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_X, q_Y + delta_Y, q_Z))
    
    B_magnitude_1 = np.sqrt(B_X_1**2 + B_Y_1**2 + B_Z_1**2)
    
    # n+2 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_2 = np.squeeze(find_B_X(q_X, q_Y + 2*delta_Y, q_Z))
    B_Y_2 = np.squeeze(find_B_Y(q_X, q_Y + 2*delta_Y, q_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_X, q_Y + 2*delta_Y, q_Z))
    
    B_magnitude_2 = np.sqrt(B_X_2**2 + B_Y_2**2 + B_Z_2**2)
    
    # n+3 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_3 = np.squeeze(find_B_X(q_X, q_Y + 3*delta_Y, q_Z))
    B_Y_3 = np.squeeze(find_B_Y(q_X, q_Y + 3*delta_Y, q_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_X, q_Y + 3*delta_Y, q_Z))
    
    B_magnitude_3 = np.sqrt(B_X_3**2 + B_Y_3**2 + B_Z_3**2)
    
    
    # Compute the value of the partial derivative using second order FFD
    d2B_dY2 = (
                ( 
                    (2)*B_magnitude_0 + (-5)*B_magnitude_1 
                    + (4)*B_magnitude_2 + (-1)*B_magnitude_3 
                ) / (delta_Y**2)
            )
    
    return d2B_dY2


def find_d2B_dZ2_FFD(q_X, q_Y, q_Z, delta_Z, find_B_X, find_B_Y, find_B_Z): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the magnitude of the 
    magnetic field, B, with respect to z using second order (delta_Z^2) 
    forward finite difference (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Z (float):
        spatial step in the z-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dZ2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the magnetic field
        with respect to z using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    # n step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_0 = np.squeeze(find_B_X(q_X, q_Y, q_Z))
    B_Y_0 = np.squeeze(find_B_Y(q_X, q_Y, q_Z))
    B_Z_0 = np.squeeze(find_B_Z(q_X, q_Y, q_Z))
    B_magnitude_0 = np.sqrt(B_X_0**2 + B_Y_0**2 + B_Z_0**2)
    
    # n+1 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_1 = np.squeeze(find_B_X(q_X, q_Y, q_Z + delta_Z))
    B_Y_1 = np.squeeze(find_B_Y(q_X, q_Y, q_Z + delta_Z))
    B_Z_1 = np.squeeze(find_B_Z(q_X, q_Y, q_Z + delta_Z))
    
    B_magnitude_1 = np.sqrt(B_X_1**2 + B_Y_1**2 + B_Z_1**2)
    
    # n+2 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_2 = np.squeeze(find_B_X(q_X, q_Y, q_Z + 2*delta_Z))
    B_Y_2 = np.squeeze(find_B_Y(q_X, q_Y, q_Z + 2*delta_Z))
    B_Z_2 = np.squeeze(find_B_Z(q_X, q_Y, q_Z + 2*delta_Z))
    
    B_magnitude_2 = np.sqrt(B_X_2**2 + B_Y_2**2 + B_Z_2**2)
    
    # n+3 step in the FFD
    #
    # np.squeeze() to remove any additional axes of size 1, see description
    # in find_H
    B_X_3 = np.squeeze(find_B_X(q_X, q_Y, q_Z + 3*delta_Z))
    B_Y_3 = np.squeeze(find_B_Y(q_X, q_Y, q_Z + 3*delta_Z))
    B_Z_3 = np.squeeze(find_B_Z(q_X, q_Y, q_Z + 3*delta_Z))
    
    B_magnitude_3 = np.sqrt(B_X_3**2 + B_Y_3**2 + B_Z_3**2)
    
    
    # Compute the value of the partial derivative using second order FFD
    d2B_dZ2 = (
                ( 
                    (2)*B_magnitude_0 + (-5)*B_magnitude_1 
                    + (4)*B_magnitude_2 + (-1)*B_magnitude_3 
                ) / (delta_Z**2)
            )
    
    return d2B_dZ2


### Mixed Derivatives of the Magnetic Field (FFD) ###


def find_d2B_dX_dY_FFD(q_X, q_Y, q_Z, 
                       delta_X, delta_Y, 
                       find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative dB/(dX dY) using the second order FFD
    approximation. Rewrites the derivative as:
        d/dX(dB/dY)
    and apply a second order FFD approximation to the above first order
    derivative (d/dX) by utilzing the find_dB_dY_FFD function.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the z-direction.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dX_dY (float or vector of shape (n,)):
        the value of the mixed derivative d^2B/(dX dY) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dB/dY in n step of the FFD
    dB_dY_0 = find_dB_dY_FFD(q_X, q_Y, q_Z, delta_Y,
                             find_B_X, find_B_Y, find_B_Z)
    
    # dB/dY in n+1 step of the FFD
    dB_dY_1 = find_dB_dY_FFD(q_X + delta_X, q_Y, q_Z, delta_Y,
                             find_B_X, find_B_Y, find_B_Z)
    
    # dB/dY in n+2 step of the FFD
    dB_dY_2 = find_dB_dY_FFD(q_X + 2*delta_X, q_Y, q_Z, delta_Y,
                             find_B_X, find_B_Y, find_B_Z)

    d2B_dX_dY = ( (-3/2)*dB_dY_0 + (2)*dB_dY_1 + (-1/2)*dB_dY_2 ) / (delta_X)
    
    return d2B_dX_dY


def find_d2B_dX_dZ_FFD(q_X, q_Y, q_Z, 
                       delta_X, delta_Z, 
                       find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative dB/(dX dZ) using the second order FFD
    approximation. Rewrites the derivative as:
        d/dX(dB/dZ)
    and apply a second order FFD approximation to the above first order
    derivative (d/dX) by utilzing the find_dB_dZ_FFD function.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the z-direction.
        
    delta_Z (float):
        spatial step in the z-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dX_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2B/(dX dZ) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dB/dZ in n step of the FFD
    dB_dZ_0 = find_dB_dZ_FFD(q_X, q_Y, q_Z, delta_Z,
                             find_B_X, find_B_Y, find_B_Z)
    
    # dB/dZ in n+1 step of the FFD
    dB_dZ_1 = find_dB_dZ_FFD(q_X + delta_X, q_Y, q_Z, delta_Z,
                             find_B_X, find_B_Y, find_B_Z)
    
    # dB/dZ in n+2 step of the FFD
    dB_dZ_2 = find_dB_dZ_FFD(q_X + 2*delta_X, q_Y, q_Z, delta_Z,
                             find_B_X, find_B_Y, find_B_Z)

    d2B_dX_dZ = ( (-3/2)*dB_dZ_0 + (2)*dB_dZ_1 + (-1/2)*dB_dZ_2 ) / (delta_X)
    
    return d2B_dX_dZ


def find_d2B_dY_dZ_FFD(q_X, q_Y, q_Z, 
                       delta_Y, delta_Z, 
                       find_B_X, find_B_Y, find_B_Z):
    
    """
    DESCRIPTION
    ==============================
    Finds the mixed derivative dB/(dY dZ) using the second order FFD
    approximation. Rewrites the derivative as:
        d/dY(dB/dZ)
    and apply a second order FFD approximation to the above first order
    derivative (d/dY) by utilzing the find_dB_dZ_FFD function.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
        
    delta_Z (float):
        spatial step in the z-direction.
        
    find_B_X (function): 
        Computes the X component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Y (function): 
        Computes the Y component of the magnetic field. Returns either a single
        float or a (n,) numpy array.
        
    find_B_Z (function): 
        Computes the Z component of the magnetic field. Returns either a single
        float or a (n,) numpy array.

    OUTPUT
    ==============================
    d2B_dY_dZ (float or vector of shape (n,)):
        the value of the mixed derivative d^2B/(dY dZ) after applying 
        second order FFD approximation. Note that if this is a vector of shape
        (n,) because the inputs were also vectors of shape (n,), then this is
        simply the value of the partial derivative at different times, 
        corresponding to the different times of the given inputs.
    """
    # dB/dZ in n step of the FFD
    dB_dZ_0 = find_dB_dZ_FFD(q_X, q_Y, q_Z, delta_Z,
                             find_B_X, find_B_Y, find_B_Z)
    
    # dB/dZ in n+1 step of the FFD
    dB_dZ_1 = find_dB_dZ_FFD(q_X, q_Y + delta_Y, q_Z, delta_Z,
                             find_B_X, find_B_Y, find_B_Z)
    
    # dB/dZ in n+2 step of the FFD
    dB_dZ_2 = find_dB_dZ_FFD(q_X, q_Y + 2*delta_Y, q_Z, delta_Z,
                             find_B_X, find_B_Y, find_B_Z)

    d2B_dY_dZ = ( (-3/2)*dB_dZ_0 + (2)*dB_dZ_1 + (-1/2)*dB_dZ_2 ) / (delta_Y)
    
    return d2B_dY_dZ

### Second Order Derivative of the poloidal flux (FFD) ###

def find_d2_poloidal_flux_dX2_FFD(q_X, q_Y, q_Z, delta_X, 
                                  interp_poloidal_flux): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the poloidal flux, 
    with respect to x using second order (delta_X^2) forward finite difference 
    (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_X (float):
        spatial step in the x-direction.
    
    interp_poloidal_flux (function):
        computes the value of the poloidal flux given position coordinates
        through interpolation. Note that in this case, the position coordinates
        are the coordinates of the vector q since we are interested in points 
        along the central ray of the beam.

    OUTPUT
    ==============================
    d2_poloidal_flux_dX2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the poloidal flux
        with respect to x using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step of the poloidal flux
    poloidal_flux_0 = interp_poloidal_flux(q_X, q_Y, q_Z)
    
    # n+1 step of the poloidal flux 
    poloidal_flux_1 = interp_poloidal_flux(q_X + delta_X, q_Y, q_Z) 
    
    # n+1 step of the poloidal flux 
    poloidal_flux_2 = interp_poloidal_flux(q_X + 2*delta_X, q_Y, q_Z)
    
    # n+1 step of the poloidal flux 
    poloidal_flux_3 = interp_poloidal_flux(q_X + 3*delta_X, q_Y, q_Z)
    
    d2_poloidal_flux_dX2 = (
                            ( (2)*poloidal_flux_0 + (-5)*poloidal_flux_1 
                             + (4)*poloidal_flux_2 + (-1)*poloidal_flux_3 )
                            / (delta_X**2)
                            )

    return d2_poloidal_flux_dX2


def find_d2_poloidal_flux_dY2_FFD(q_X, q_Y, q_Z, delta_Y, 
                                  interp_poloidal_flux): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the poloidal flux, 
    with respect to y using second order (delta_Y^2) forward finite difference 
    (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Y (float):
        spatial step in the y-direction.
    
    interp_poloidal_flux (function):
        computes the value of the poloidal flux given position coordinates
        through interpolation. Note that in this case, the position coordinates
        are the coordinates of the vector q since we are interested in points 
        along the central ray of the beam.

    OUTPUT
    ==============================
    d2_poloidal_flux_dY2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the poloidal flux
        with respect to y using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step of the poloidal flux
    poloidal_flux_0 = interp_poloidal_flux(q_X, q_Y, q_Z)
    
    # n+1 step of the poloidal flux 
    poloidal_flux_1 = interp_poloidal_flux(q_X, q_Y + delta_Y, q_Z) 
    
    # n+1 step of the poloidal flux 
    poloidal_flux_2 = interp_poloidal_flux(q_X, q_Y + 2*delta_Y, q_Z)
    
    # n+1 step of the poloidal flux 
    poloidal_flux_3 = interp_poloidal_flux(q_X, q_Y + 3*delta_Y, q_Z)
    
    d2_poloidal_flux_dY2 = (
                            ( (2)*poloidal_flux_0 + (-5)*poloidal_flux_1 
                             + (4)*poloidal_flux_2 + (-1)*poloidal_flux_3 )
                            / (delta_Y**2)
                            )

    return d2_poloidal_flux_dY2


def find_d2_poloidal_flux_dZ2_FFD(q_X, q_Y, q_Z, delta_Z, 
                                  interp_poloidal_flux): 
    """
    DESCRIPTION
    ==============================
    Computes the second order partial derivative of the poloidal flux, 
    with respect to z using second order (delta_Z^2) forward finite difference 
    (FFD) approximation.

    INPUT
    ==============================
    q_X (float or vector of shape (n,)):
        X - component of q. Can be a single number or a vector indicating the
        value of q_X at different times.
        
    q_Y (float or vector of shape (n,)):
        Y - component of q. Can be a single number or a vector indicating the
        value of q_Y at different times.
        
    q_Z (float or vector of shape (n,)):
        Z - component of q. Can be a single number or a vector indicating the
        value of q_Z at different times.
        
    delta_Z (float):
        spatial step in the z-direction.
    
    interp_poloidal_flux (function):
        computes the value of the poloidal flux given position coordinates
        through interpolation. Note that in this case, the position coordinates
        are the coordinates of the vector q since we are interested in points 
        along the central ray of the beam.

    OUTPUT
    ==============================
    d2_poloidal_flux_dZ2 (float or or vector of shape (n,)):
        the value of the second order partial derivative of the poloidal flux
        with respect to z using second order FFD. Note that if this is a vector 
        of shape (n,) because the inputs were also vectors of shape (n,),
        then this is simply the value of the partial derivative at
        different times, corresponding to the different times of the given 
        inputs.
    """  
    
    # n step of the poloidal flux
    poloidal_flux_0 = interp_poloidal_flux(q_X, q_Y, q_Z)
    
    # n+1 step of the poloidal flux 
    poloidal_flux_1 = interp_poloidal_flux(q_X, q_Y, q_Z + delta_Z) 
    
    # n+1 step of the poloidal flux 
    poloidal_flux_2 = interp_poloidal_flux(q_X, q_Y, q_Z + 2*delta_Z)
    
    # n+1 step of the poloidal flux 
    poloidal_flux_3 = interp_poloidal_flux(q_X, q_Y, q_Z + 3*delta_Z)
    
    d2_poloidal_flux_dZ2 = (
                            ( (2)*poloidal_flux_0 + (-5)*poloidal_flux_1 
                             + (4)*poloidal_flux_2 + (-1)*poloidal_flux_3 )
                            / (delta_Z**2)
                            )

    return d2_poloidal_flux_dZ2




########## FUNCTIONS FOR LAUNCH ANGLE  ##########
# IGNORE FOR NOW (dont use mirrors, just give poloidal/toroidal launch angles)
#
# Written by Neal Crocker, added to Scotty by Valerian.
#
# Converts mirror angles of the MAST DBS to launch angles (genray)
#
# Genray -> Scotty/Torbeam: poloidal and toroidal launch angles 
# have opposite signs












"""
DESCRIPTION
==============================


MODIFICATIONS FROM VALERIANS ORIGINAL CODE
==============================


INPUT
==============================


OUTPUT
==============================

"""






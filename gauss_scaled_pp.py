import numpy as np

#2nd arg: the scales
#returns a numpy array of positions of pivot rows

#def order_by_pivotscaled(A, s, n):
#finds pivot row
#arg1: mtx_w_key is sent
#returns key of pivot row, which is a string
def find_pivot_row(mtx_w_key, sliced_A, s, n):
    #only need to know n-1 rows; last row will be appended last
###    pivot_row_pos = np.zeros(n-1) #has n-1 zeros to be replaced
###    for j in range(n-1): #replaces n-1 els in pivot_row_pos

    #slicing doesn't alter original mtx
    #keep all rows; ignores col0(key col), keeps col1 onwards
    A = mtx_w_key[:, 1:] ##FIXME
    #            row,col

 

    pivot_row_pos = 0
    for i in range(n-1): #n=3: i+1 = 2 => i = 1 =>n-1
        if (A[i][0] / s[i]) < (A[i+1][0] / s[i+1]):
            pivot_row_pos = i + 1
        ####    pivot_row_pos[j] = i+1

       #ref: https://stackoverflow.com/questions/40382384/finding-a-matching-row-in-a-numpy-matrix
    #(a==(1,3)).all(axis=1) returns a bool col: True if (1,3) found in a
    # equiv: 
    #returns a col of T/F where T==the row where A[] matches the mtx
    #(mtx_w_key==A[pivot_row_pos]).all(axis=1)
    bool_col_mtx = (mtx_w_key==A[pivot_row_pos]).all(axis=1)
    
    for i in range(len(bool_col_mtx)):
        if (bool_col_mtx[i]):
            #if the row==True, returns the key of that row
            return mtx_w_key[i][0]
    


#finds the scales
#arg: matrix array
#returns 1xn array
def det_scales(A):
    n = len(A) #nxn square mtx
    #scales (for each row)
    s = np.zeros(n) # 1xn array filled with float zeros
    for i in range(n):
        #for each row in A: abs, then max
        s[i] = max(np.absolute(A[i]))
    return s

def row_wo_key(mtx, dict1, key):
#dict1[key]: returns idx that corresponds with key
#                         1: keeps all cols 
#FIXME rm    idx_of_key = dict1[key]
    return mtx[dict1[key], 1:]
    #1st arg:      ^{refers to row with this idx} like mtx[1][1:]

#args: A = mtx (square)
#       b = col mtx
#       n = # of cols := len(A) or len(b)
def guass_scaled_partial_pivot(A, b):
    n = len(A) #nxn square mtx

    #determines the scales(1xn array)
    s = det_scales(A)

    #appends a key to first col of mtxA so that each row has a key
    #FIXME: if nxn, need n keys
    keys = ['first', 'second', 'third']
    #inits a numpy array w/ vals 0 to n-1
    vals = np.zeros((n,1))
    for i in range(n):
        vals[i] = i
    print("vals: ", vals)
    print(A)
    dict1 = dict(zip(keys, vals))

    #stacks column wise
    mtx_w_key = np.hstack((vals, A))  
    print("mtx_w_key: ")
    print(mtx_w_key) 

    #1st iteration: considers 1st pivot row in nxn
    #2nd iteration: considers 2nd pivot row in (n-1)x(n-1)...etc...

    #choose order of pivot rows depending on pivot position / scaled
    #will store new mtx's rows in each element pos of this array
    ordered_by_scale_mtx = np.zeros(n)

    
    list_row_indices = []
    [list_row_indices.append(i) for i in range(n)]

    #print(find_pivot_row(A, s, n)) #fixme: rm

    #ordering the matrix by pivot scaled
    # 3 rows would only need 2 comparisons => n-1 comparisons
    for i in range(0, n-1): #3: [i:0, 1]

        #find_pivot_row returns the str key
        #A == a sliced version of A (also w/o key)
        pivot_key = find_pivot_row(mtx_w_key, A, s, n)
  ##FIXME rm?      #swap the row with the pivot index to 
    #ordered_by_scale_mtx[i] = A[pivot_index]

        idx_of_key = dict1[key]
        #inserts row_wo_key into new mtx
        ordered_by_scale_mtx[i] = row_wo_key(mtx_w_key, dict1, pivot_key)
        #removes pivot_index from list
        list_row_indices.remove(idx_of_key)
        #delete pivot row from A
            #3rd arg: axis=0 == row
        A = np.delete(A, idx_of_key, 0)
        #delete first col
            #3rd arg: axis=1 == col
        A = np.delete(A, 0, axis=1)

    #append last row
    #only one num left in list_row_indices
    np.append(ordered_by_scale_mtx, A[list_row_indices[0]])
#todo: return a float with 4 sig figs(project brief)
    return ordered_by_scale_mtx

def print_mtx_array(mtx):
    for i in range(len(mtx)):
        print(mtx[i])

def main():
    B = [[1,2,3],[5,6,7],[8,9,10]] #1st pivot row: 2=row3
    C = np.array([[2,3,0],[-1,2,-1],[3,0,2]]) #1st pivot row: 2=row3
    #s = 4,5,7; .75,.85 => 1st pivot row = 2=row3
    D = [[3,4,3],[1,5,-1],[6,3,7]]
    b = [1,2,3]
    mtx = guass_scaled_partial_pivot(C, b)

    #print_mtx_array(mtx)

    return

main()


import numpy as np
from math import log10, floor #used for rounding w.r.t sig figs

#2nd arg: the scales
#returns a numpy array of positions of pivot rows

#def order_by_pivotscaled(A, s, n):
#finds pivot row
#arg1: mtx_w_key is sent
#returns pivot row's index
def get_pivot_row_idx(mtx_w_key, sliced_A, s_w_key):
    #only need to know n-1 rows; last row will be appended last
###    pivot_row_pos = np.zeros(n-1) #has n-1 zeros to be replaced
###    for j in range(n-1): #replaces n-1 els in pivot_row_pos

    #slicing doesn't alter original mtx
    #keep all rows; ignores col0(key col), keeps col1 onwards
###    A = mtx_w_key[:, 1:] ##FIXME
    #            row,col
    print("s_w_key should be --1 each it: ", s_w_key)
 

    pivot_row_pos = 0
    for i in range(len(sliced_A) - 1): #n=3: i+1 = 2 => i = 1 =>n-1
        if (sliced_A[i][0] / s_w_key[i][1]) < (sliced_A[i+1][0] / s_w_key[i+1][1]):
            pivot_row_pos = i + 1
        ####    pivot_row_pos[j] = i+1

       #ref: https://stackoverflow.com/questions/40382384/finding-a-matching-row-in-a-numpy-matrix
    #(a==(1,3)).all(axis=1) returns a bool col: True if (1,3) found in a
    # equiv: 
    #returns a col of T/F where T==the row where A[] matches the mtx
    #(mtx_w_key==A[pivot_row_pos]).all(axis=1)
###    bool_col_mtx = (mtx_w_key==A[pivot_row_pos]).all(axis=1)
    
    #a mtx of bools where 
    print("A's pivot row: ", sliced_A[pivot_row_pos]) #fixme: checks out
    print('sliced_A: ', sliced_A)
    
###    bool_mtx = np.isin(mtx_w_key[:,1:], sliced_A[pivot_row_pos])
#    print(bool_mtx)
 #   bool_col_mtx = np.all(bool_mtx, axis=1)
  #  print(bool_col_mtx) #FIXME doesn't check out


    slice_pos = len(mtx_w_key) - len(sliced_A)
    print("1st slice_pos should be 0, ->+1 == 1 in where_s_isin: ", slice_pos)
    print(">where_sliced_isin should return row_idx: ", where_sliced_isin(mtx_w_key, sliced_A[pivot_row_pos], slice_pos))
    return where_sliced_isin(mtx_w_key, sliced_A[pivot_row_pos], slice_pos)
    #checking if sliced_A's pivot row is in mtx_w_key
    #np.where(c == C)[0][0] #gives row idx where c is found in C
####    print("pivot row should be two: ", np.where(sliced_A[pivot_row_pos] == mtx_w_key)[0][0])
####    return np.where(sliced_A[pivot_row_pos] == mtx_w_key[:, sliced_pos:])[0][0]

 #   for i in range(len(bool_col_mtx)):
  #      if (bool_col_mtx[i]):
   #         #if the row==True, returns the pivot row index
    #        print("pivot row idx: ", mtx_w_key[i][0] )
     #       return mtx_w_key[i][0]
    
def where_sliced_isin(mtx_w_key, sliced_A_row, slice_pos):
 #   #working backwards
  #  for y in range(len(sliced_A)):
   #     for x in range(len(sliced_A)): #skipping idx col
    #        if sliced_A[y][x]
 ###   print("orig mtx_w_key: ", mtx_w_key)
    print(">in where_sliced_isin: should be [3 0 2]: >>", sliced_A_row)
###    print(" >>mtx_w_key but w/o key >>", mtx_w_key[:, slice_pos+1:])
###    for idx in range(len(mtx_w_key)):
###        if (sliced_A_row in mtx_w_key[:, slice_pos+1:]): #+1 to account for idx col
###            return idx
    return np.where(sliced_A_row == mtx_w_key[:, slice_pos+1:])[0][0]


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

#FIXME don't seem to need this def
def row_wo_key(mtx, dict1, key):
#dict1[key]: returns idx that corresponds with key
#                         1: keeps all cols 
#FIXME rm    idx_of_key = dict1[key]
    return mtx[dict1[key], 1:]
    #1st arg:      ^{refers to row with this idx} like mtx[1][1:]

#concats array A with array b
def aug_mtx(A, b):
    aug_mtx = np.hstack((A,b))
    print(aug_mtx)
    return aug_mtx

#arg: must take in augmented mtx
def forward_elim(aug_mtx):
    return

#args: A = mtx (square)
#       b = col mtx
#       n = # of cols := len(A) or len(b)
def guass_scaled_partial_pivot(A, b):
    n = len(A) #nxn square mtx

    #will insert b's values ordered by scales in here
    b_ordered = np.zeros((n,1))
    #determines the scales(1xn array)
    s = det_scales(A)
    s_col_vect = np.zeros((n,1))
    for i in range(len(s)):
        s_col_vect[i] = s[i]

    #appends a key to first col of mtxA so that each row has a key
    #FIXME: if nxn, need n keys
    keys = ['first', 'second', 'third']
    #inits a numpy array w/ vals 0 to n-1
    vals = np.zeros((n,1))
    for i in range(n):
        vals[i] = i

    

    #{'first': array([0.]), 'second': array([1.]), 'third': array([2.])}
    dict1 = dict(zip(keys, vals))

    #stacks column wise
    mtx_w_key = np.hstack((vals, A))
    s_w_key = np.hstack((vals, s_col_vect))  
    

    #1st iteration: considers 1st pivot row in nxn
    #2nd iteration: considers 2nd pivot row in (n-1)x(n-1)...etc...

    #choose order of pivot rows depending on pivot position / scaled
    #will store new mtx's rows in each element pos of this array
    ordered_by_scale_mtx = np.zeros((n,n))

    
    list_row_indices = []
    [list_row_indices.append(i) for i in range(n)]

    #print(get_pivot_row_idx(A, s, n)) #fixme: rm

    #ordering the matrix by pivot scaled
    # 3 rows would only need 2 comparisons => n-1 comparisons
    for j in range(0, n-1): #3: [i:0, 1]

        #get_pivot_row_idx returns the str key
        #A == a sliced version of A (also w/o key)
        pivot_idx = get_pivot_row_idx(mtx_w_key, A, s_w_key)
        print("@@@@@@@@@pivot_idx: ", pivot_idx)
  ##FIXME rm?      #swap the row with the pivot index to 
    #ordered_by_scale_mtx[i] = A[pivot_index]

 ###       print("dict1[key] = ", dict1[pivot_key])
 ###       idx_of_key = dict1[pivot_key]
        #inserts row_wo_key into new mtx
###        ordered_by_scale_mtx[i] = row_wo_key(mtx_w_key, dict1, pivot_row)
        #              the row with this pivot idx, ignoring the idx col
        ordered_by_scale_mtx[j] = mtx_w_key[pivot_idx, 1:]
        #removes pivot_index from list
        list_row_indices.remove(pivot_idx)
   #TODO     #rm the s that corresponds with the pivot_idx so it doesn't affect get_pivot
        print("        >{} >>>>>>>>pivot idx: {}".format(j, pivot_idx))
        print("s_w_key b4 deletion of pivot: ", s_w_key)
        if (len(s_w_key) != 1):
            for i in range(len(s_w_key)): #fixme? might be len issue 
                if s_w_key[i][0] == pivot_idx:
                    print("beginning deletion of pivot in s_w_key")
                    s_w_key = np.delete(s_w_key, i, 0)
                    print("idx and s should be deleted in s_w_key: ", s_w_key)
                    break
        #delete pivot row from A
            #3rd arg: axis=0 == row
        A = np.delete(A, pivot_idx, 0)
        #delete first col
            #3rd arg: axis=1 == col
        A = np.delete(A, 0, axis=1)

        b_ordered[j] = b[pivot_idx]

    
#    print("      >>>>>>last row idx should be 1: >>", list_row_indices[0])
#    print("mtx that is mising the last row:", ordered_by_scale_mtx)
    #append last row
    #only one num left in list_row_indices
    ordered_by_scale_mtx[n-1] = mtx_w_key[list_row_indices[0], 1:]
    b_ordered[n-1] = b[list_row_indices[0]]

#TODO: return a float with 4 sig figs(project brief)
    print("          >>>>>>>>>>>>> final ordered mtx: ", ordered_by_scale_mtx)
    print("         ____________ordered::scaled augmented mtx: ")
    aug_mtx = np.hstack((ordered_by_scale_mtx, b_ordered))
    return aug_mtx

def print_mtx_array(mtx):
    for i in range(len(mtx)):
        print(mtx[i])


#rounds a val w.r.t sig figs
def round_w_sig(x, sig):
    #                             vv takes care of negative nums
    return round(x, (sig) - int(floor(log10(abs(x))))-1)

def main():
    #examples:
    B = [[1,2,3],[5,6,7],[8,9,10]] #1st pivot row: 2=row3
    C = np.array([[2,3,0],[-1,2,-1],[3,0,2]]) #1st pivot row: 2=row3
    #s = 4,5,7; .75,.85 => 1st pivot row = 2=row3
    D = [[3,4,3],[1,5,-1],[6,3,7]]
    b = [8,0,9]

    #ref: lecture 6 and 7 slides
    lec6 = np.array([[2, 10000], 
                    [1,1]])
    lec6_b = [[10000], 
            [2]]
    
#####    mtx = guass_scaled_partial_pivot(C, b)

    #this produces the correct order of the rows as shown in lec6n7 slides
    #returns an augmented mtx of correctly ordered rows
    mtx = guass_scaled_partial_pivot(lec6, lec6_b)
    print("lec6's augmented matrix: ", mtx)


    #*one line code for finding solution*:
    #print(np.linalg.solve(C,b))
    print("exact sol = [[1],[1]]\n numpy sol: ", np.linalg.solve(lec6, lec6_b))

    #convert numpy to a native python type using .item()
    sol_w_4_sig_figs = []
    #FIXME change      v 1
    for i in range(len(lec6_b)):
        #FIXME change             v 2   v 3end
        tmp_val = np.linalg.solve(lec6, lec6_b)[i].item()
        #rounding a val w/ 4 sig figs 
        sol_w_4_sig_figs.append(round_w_sig(tmp_val, 4))
    print(sol_w_4_sig_figs)


    return

main()


'''
Machine Learning Practice
Practice 1.2: Hypothesis counting
Author: Polly Zhou
Institution: Tsinghua University TEEP

Thanks to 'wottzh': https://zhuanlan.zhihu.com/p/355235881

Estimate the number of hypothesises.
'''
import numpy as np
import itertools as iter
import time

# Turn a 3-D array into an 18-D array
# Input: 1x3 array, [color, root, sound], 0 for "*"
# Output: 1x18 array, [18 hypothesises], 0 for "x", 1 for "√"
def get18(input_):
    output_ = np.zeros([2,3,3])
    if input_[0] == 0:                  # Mark the "1" position for color
        color_ = [0, 1]
    else:
        color_ = [input_[0] - 1]
    if input_[1] == 0:                  # Mark the "1" position for root
        root_ = [0, 1, 2]
    else:
        root_ = [input_[1] - 1]
    if input_[2] == 0:                  # Mark the "1" position for sound
        sound_ = [0, 1, 2]
    else:
        sound_ = [input_[2] - 1]
    for i1 in color_:
        for i2 in root_:
            for i3 in sound_:
                output_[i1][i2][i3] = 1
    output_ = output_.flatten()
    return output_


# For convenience, we correspond 0-47 to each array, here we use MOD
# 0 for "*"
def turn48array(num):
    i = num // 16             # 1: green, 2: black
    j = num % 16 // 4         # 1: curl, 2: little curl, 3: straight
    k = num % 16 % 4          # 1: clunk, 2: clear, 3: dull                                                     
    result = [i, j, k]
    return result

def turn48to18(n):
    temp = turn48array(n)
    output = get18(temp)
    return output


# Get k arrays
# For convenience, we pick k numbers and get 18-arrays
# Put arrays together, and get all the 1-existing positions to avoid ovealap
def hypospace(k):
    hypo_list=[]
    for i in iter.combinations(range(48),k):
        subset = 0
        for j in range(k):
            p = i[j]
            subset = subset | hypo48_oct[p]
        hypo_list.append(subset)
        if len(hypo_list) > 5000000:
            hypo_list = list(set(hypo_list))  
    hypo_list = list(set(hypo_list))
    print("正好包含%d个合取式的析合范式能表示 : %d 种假设" % (k, len(hypo_list)))
    return(hypo_list)

hypo48 = []
for i in range(48):
    h = turn48to18(i)
    hypo48.append(h)
hypo48_oct = []
for hypo in hypo48:
    hypo_bin = '0b'
    for i in hypo:
        hypo_bin = hypo_bin + str(int(i))
    hypo_oct = eval(hypo_bin)
    hypo48_oct.append(hypo_oct)

start = time.time()
hypo_stack = []
for k in range(18):
    k = k+1
    hypo_list = hypospace(k)
    hypo_stack = hypo_stack + hypo_list
    hypo_stack = list(set(hypo_stack))
    end = time.time()
    print( "最多包含%d个合取式的析合范式能表示 : %d 种假设"%(k,len(hypo_stack)+1))  # Add null set
    print("用时：%.2fs\n"%(end-start))

# 01 knapsack memoization

def knapsack(wt_arr,val_arr,capacity,n):
    our_arr = [[-1 for i in range(capacity+1)] for j in range(n+1)] # memoization step
    if (n==0 or capacity == 0): return 0 # base case
    if our_arr[n][capacity] != -1: return our_arr[n][capacity]
    if wt_arr[n-1] <= capacity:
        our_arr[n][capacity] =  max(val_arr[n-1]+knapsack(wt_arr,val_arr,capacity-wt_arr[n-1],n-1),knapsack(wt_arr,val_arr,capacity,n-1))# val_arr[n-1]+knapsack(wt_arr,val_arr,capacity-wt_arr[n-1],n) for unbdd knapsack
        return our_arr[n][capacity]
    else:
        our_arr[n][capacity] =  knapsack(wt_arr,val_arr,capacity,n-1)
        return our_arr[n][capacity]
    
#topdown dp

def knapsack(wt_arr,val_arr,capacity,n):
    our_arr = [[-1 for i in range(capacity+1)] for j in range(n+1)]
    for i in range(capacity+1):
        our_arr[0][i] = 0
    for i in range(n+1):
        our_arr[i][0] = 0
    for i in range(1,n+1):
        for j in range(1,capacity+1):
            if wt_arr[i-1] <= j:
                our_arr[i][j] = max(val_arr[i-1] + our_arr[i-1][j-wt_arr[i-1]], our_arr[i-1][j])# val_arr[i-1] + our_arr[i][j-wt_arr[i-1]] for unbdd knapsack
            else: our_arr[i][j] = our_arr[i-1][j]
            
    return our_arr[n][capacity]

# test case      
wts = [2,5,8,6,2]
vals = [6,10,4,0,1]
capacity = 9
n = 5
print(knapsack(wts,vals,capacity,n))

# memoized subset sum from knapsack

# if there is only one arr given think of it as weight arr and capacity the max constraint given

def subsetsum(arr,target):
    n = len(arr)
    memo = [[-1 for i in range(target+1)] for j in range(n+1)]
    if (target == 0 or arr == []): return True
    if memo[n][target] != -1: return memo[n][target]
    if arr[n-1] <= target:
        memo[n][target] = subsetsum(arr[:n-1],target) or subsetsum(arr[:n-1],target-arr[n-1])
        return memo[n][target]
    else:
        memo[n][target] = subsetsum(arr[:n-1],target)
        return memo[n][target]
  
# test case      
arr = [4,2,7,1,5]
target = 11
print(subsetsum(arr,target))

# count the number of soln in the subset sum
# this probelm can be used to find the num of subsets with a given diff. s1-s2 = diff and s1 + s2 = sum(arr). now add these to eqns and use countsubsetsum(arr,diff+sum(arr) // 2)

def countsubsetsum(arr,target):
    n = len(arr)
    memo = [[-1 for i in range(target+1)] for j in range(n+1)]
    for j in range(target+1):
        memo[0][j] = 0
    for i in range(n+1):
        memo[i][0] = 1
    if memo[n][target] != -1: return memo[n][target]
    if arr[n-1] <= target:
        memo[n][target] = countsubsetsum(arr[:n-1],target) + countsubsetsum(arr[:n-1],target-arr[n-1])
        return memo[n][target]
    else:
        memo[n][target] = countsubsetsum(arr[:n-1],target)
        return memo[n][target]
  
# test case      
arr = [2,3,5,7,8,10]
target = 10
print(countsubsetsum(arr,target))

# for the minsubsetdiff to work we need the last row of the dp table of the subset sum problem. So that code need to be modified a little bit.
def subsetsum(arr,target):
    n = len(arr)
    memo = [[-1 for j in range(target+1)] for i in range(n+1)]
    for i in range(target+1):
        memo[0][i] = False
    for i in range(n+1):
        memo[i][0] = True
    for i in range(1,n+1):
        for j in range(1,target+1):
            if arr[i-1] <= j:
                memo[i][j] = memo[i-1][j] or memo[i-1][j-arr[i-1]]# the second summand will be memo[i][j-arr[i-1]] for unbdd subset sum and val[i-1] to be added if there is a val_arr
            else: memo[i][j] = memo[i-1][j]
    vec = [memo[n][i] for i in range(target+1)]
    return vec

arr = [2,4,5,7]
target = 7
print(subsetsum(arr,target))
#[True, False, True, False, True, True, True, True]

# Now use this modified code to write the main function
def minsumdiff(arr):
    n = len(arr)
    req_range = sum(arr)
    minim = req_range
    vec = subsetsum(arr,req_range//2 + 1)
    for i in range(len(vec)):
        if vec[i] == True:
            minim = min(minim,req_range-2*i)
    return abs(minim)

def rodcutting(length_arr,price_arr,rod_len):
    return unbddknapsack(wt_arr,val_arr,capacity)

def num_of_ways_coinchange(coin_arr,val):
    return unbddsubsetsum(arr = coin_arr,target = val)








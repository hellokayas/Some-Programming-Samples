# 01 knapsack memoization

def knapsack(wt_arr,val_arr,capacity,n):
    our_arr = [[-1 for i in range(capacity+1)] for j in range(n+1)] # memoization step
    if (n==0 or capacity == 0): return 0 # base case
    if our_arr[n][capacity] != -1: return our_arr[n][capacity]
    if wt_arr[n-1] <= capacity:
        our_arr[n][capacity] =  max(val_arr[n-1]+knapsack(wt_arr,val_arr,capacity-wt_arr[n-1],n-1),knapsack(wt_arr,val_arr,capacity,n-1))
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
                our_arr[i][j] = max(val_arr[i-1] + our_arr[i-1][j-wt_arr[i-1]], our_arr[i-1][j])
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


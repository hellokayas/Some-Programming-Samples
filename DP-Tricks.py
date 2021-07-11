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

# test case      
wts = [2,5,8,6,2]
vals = [6,10,4,0,1]
capacity = 9
n = 5
print(knapsack(wts,vals,capacity,n))

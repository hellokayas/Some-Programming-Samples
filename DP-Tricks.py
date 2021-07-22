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

def mincoin(coin_arr,total):# the min number of coins to be used from the coin_arr to make the given total
    n = len(coin_arr)
    memo = [[2**30 for j in range(total+1)] for i in range(n+1)]
    for i in range(n+1):
        memo[i][0] = 0
    for i in range(total+1):# this means not possible, mathematically we set this to infinity
        memo[0][i] = 2**31-1
    for i in range(1,total+1):# this is the case where we have to initialize the second row too
        if (i % coin_arr[0] == 0):# if the first num in the arr divides the total, then we can use those many coins
            memo[1][i] = i // coin_arr[0]
        else: memo[1][i] = 2**31-1# this means not possible, mathematically we set this to infinity
    for i in range(n+1):
        for j in range(total + 1):
            if coin_arr[i-1] <= j:
                memo[i][j] = min(1+memo[i][j-coin_arr[i-1]], memo[i-1][j])# the unbddknapsack
            else: memo[i][j] = memo[i-1][j]
    return memo[n][total]

# test case
coin_arr = [1,2,3,4]
total = 297
print(mincoin(coin_arr,total))

def lcsubstr(p,q,m,n):
    memo = [[0 for i in range(n+1)] for j in range(m+1)]
    result = 0
    for i in range(m+1):
        memo[i][0] = 0
    for i in range(n+1):
        memo[0][i] = 0
    for i in range(1,m+1):
        for j in range(1,n+1):
            if p[i-1] == q[j-1]:
                memo[i][j] = 1 + memo[i-1][j-1]
                result = max(result,memo[i][j])
            else:
                memo[i][j] = 0 #max(memo[i-1][j],memo[i][j-1])
    return result

#test case
p = "abhtgj"
q = "abhtokjojkjnv"
m,n = len(p),len(q)
print(lcs(p,q,m,n))

# Printing the  LCS now

def printlcs(p,q,m,n):
    memo = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if p[i-1]==q[j-1]: memo[i][j] = 1 + memo[i-1][j-1]
            else: memo[i][j] = max(memo[i-1][j],memo[i][j-1])
    #print(memo)            
    i,j = m,n
    string = ""
    while(i>0 and j>0):
        if p[i-1] == q[j-1]:
            string += p[i-1]
            i -= 1
            j -= 1
            
            #print(string)
        else:
            if memo[i-1][j] > memo[i][j-1]:
                i = i-1
            else: j = j-1
    return string[::-1]

#test case
p = "abhtgjxyuhifluwbhflib"
q = "abhtokjojkjnvxyuibfulwabfaulwb"
m,n = len(p),len(q)
print(printlcs(p,q,m,n))

# Minimum Number of Insertion and Deletion to convert String a to String b
def min_num(a,b):# if replace operation is also allowed, then the problem becomes the famous EDIT DISTANCE problem
    x,y = len(a),len(b)
    delete = x - lcs(a,b,x,y)
    insert = y - lcs(a,b,x,y)
    return [delete,insert]

# Longest Palindromic Subsequence
def LPS(a):
    return lcs(a,a[::-1])
# Minimum number of deletion in a string a to make it a palindrome
# this will be len(a)-LPS(a)


# printing the longest common superseq

def printscs(p,q,m,n):
    memo = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if p[i-1]==q[j-1]: memo[i][j] = 1 + memo[i-1][j-1]
            else: memo[i][j] = max(memo[i-1][j],memo[i][j-1])
    #print(memo)            
    i,j = m,n
    string = ""
    while(i>0 and j>0):
        if p[i-1] == q[j-1]:
            string += p[i-1]
            i -= 1
            j -= 1
            # upto this is exactly same as printing lcs
            #print(string)
        else:
            if memo[i-1][j] > memo[i][j-1]:
                string += p[i-1]# even when we move towards the maximum and the chars do not match we have to include this in the superseq
                i = i-1
            else:
                string += q[j-1]# same as the last comment
                j = j-1
        while (i>0):# either of i or j might be still non empty, so include that remaining part of the str in the superseq
            string += p[i-1]
            i = i-1
        while (j>0):# same as thelast comment
            string += q[j-1]
            j = j-1
    return string[::-1]

# Longest repeating subsequence -- exactly same code as lcs, just i != j when finding lcs of the same string.
def printlcs(s):
    p,q,m,n = s,s,len(s),len(s)
    memo = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if p[i-1]==q[j-1] and i != j: memo[i][j] = 1 + memo[i-1][j-1]
            else: memo[i][j] = max(memo[i-1][j],memo[i][j-1])
    
    return memo[m][n]






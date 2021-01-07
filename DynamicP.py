import math
import sys
import string
from collections import Counter
from collections import defaultdict
from collections import deque
from bisect import bisect_left
'''
Given two strings A and B. Find the longest common sequence ( A sequence which does not need to be contiguous), which is common in both
 the strings.
You need to return the length of such longest common subsequence.
Soln : 
Let the input sequences be X[0..m-1] and Y[0..n-1] of lengths m and n respectively. And let L(X[0..m-1], Y[0..n-1]) be the length of LCS of the two sequences X and Y. Following is the recursive definition of L(X[0..m-1], Y[0..n-1]).

If last characters of both sequences match (or X[m-1] == Y[n-1]) then
L(X[0..m-1], Y[0..n-1]) = 1 + L(X[0..m-2], Y[0..n-2])

If last characters of both sequences do not match (or X[m-1] != Y[n-1]) then
L(X[0..m-1], Y[0..n-1]) = MAX ( L(X[0..m-2], Y[0..n-1]), L(X[0..m-1], Y[0..n-2]) )'''
def lcs(X , Y): 
    # find the length of the strings 
    m = len(X) 
    n = len(Y) 
  
    # declaring the array for storing the dp values 
    L = [[None]*(n+1) for i in xrange(m+1)] 
  
    """Following steps build L[m+1][n+1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j] , L[i][j-1]) 
  
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n] 
'''
Given a string A, find the common palindromic sequence ( A sequence which does not need to be contiguous and is a pallindrome), 
which is common in itself.
You need to return the length of longest palindromic subsequence in A.
Example Input
Input 1:
 A = "bebeeed"

Example Output
Output 1:
 4
Example Explanation
Explanation 1:
The longest common pallindromic subsequence is "eeee", which has a length of 4
Soln approach: Let X[0..n-1] be the input sequence of length n and L(0, n-1) be the length of the longest palindromic subsequence of
 X[0..n-1].
If last and first characters of X are same, then L(0, n-1) = L(1, n-2) + 2.
Else L(0, n-1) = MAX (L(1, n-1), L(0, n-2)). we can apply algorithm of finding longest common subsequence
taking input as first string and reverse of input as second string.'''
def LCPalindrome(s):
	s1 = s[::-1]
	return lcs(s,s1)
'''
Given two strings A and B, find the minimum number of steps required to convert A to B. (each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character
Soln Appraoch: We are trying to modify S1 to become S2.

We look at the first character of both the strings.
If they match, we can look at the answer from remaining part of S1 and S2.
If they don’t, we have 3 options.
1) Insert S2’s first character and then solve the problem for remaining part of S2, and S1.
2) Delete S1’s first character and trying to match S1’s remaining string with S2.
3) Replace S1’s first character with S2’s first character in which case we solve the problem for remaining part of S1 and S2.'''
def minDistance(A,B):
	n = len(A) + 1
	m = len(B) + 1
	array = [[0]*m for _ in range(n)]#arr[x][y] contains the ans for len(A) = x and len(B) = y
	for i in range(m):# so when one of the string has 0 length the num of opn is same as the length of other str
		array[0][i] = i
	for i in range(n):# same as above
		array[i][0] = i
	for i in range(n):# now the soln approach written in code
		for j in range(m):
			c = 0 if A[i-1] == B[i-1] else 1
			array[i][j] = min(array[i-1][j],array[i][j-1],array[i-1][j-1] + c)
	return array[-1][-1]# returning the last elem
'''
Given a string A, find length of the longest repeating sub-sequence such that the two subsequence don’t have same string character at
 same position, i.e., any i’th character in the two subsequences shouldn’t have the same index in the original string.
NOTE: Sub-sequence length should be greater than or equal to 2.
Soln Approach: This problem is just the modification of Longest Common Subsequence problem. The idea is to find the LCS(str, str)
 where str is the input string with the restriction that when both the characters are same, they shouldn’t be on the same index in
  the two strings.'''
def anytwo(A):
        
        n=len(A)
        if n==0:
            return 0
        
        x=[[-1]*(n+1) for i in range(n+1)]
        
        for i in range(n+1):
            for j in range(n+1):
                if i==0 or j==0 or i==j:
                    x[i][j]=0
                elif A[i-1]==A[j-1]:
                    x[i][j]=x[i-1][j-1]+1
                else:
                    x[i][j]=max(x[i-1][j],x[i][j-1])
                    
                if x[i][j]>=2:
                    return 1
        return 0
'''
Given two sequences A, B, count number of unique ways in sequence A, to form a subsequence that is identical to the sequence B.
Subsequence : A subsequence of a string is a new string which is formed from the original string by deleting some (can be none)
 of the characters without disturbing the relative positions of the remaining characters.
  (ie, “ACE” is a subsequence of “ABCDE” while “AEC” is not).
Soln Approach: 
s a typical way to implement a dynamic programming algorithm, we construct a matrix dp, where each cell dp[i][j] represents the number
 of solutions of aligning substring T[0..i] with S[0..j];

Rule 1). dp[0][j] = 1, since aligning T = “” with any substring of S would have only ONE solution which is to delete all characters in S.

Rule 2). when i > 0, dp[i][j] can be derived by two cases:

case 1). if T[i] != S[j], then the solution would be to ignore the character S[j] and align substring T[0..i] with S[0..(j-1)].
 Therefore, dp[i][j] = dp[i][j-1].

case 2). if T[i] == S[j], then first we could adopt the solution in case 1), but also we could match the characters T[i] and S[j]
 and align the rest of them (i.e. T[0..(i-1)] and S[0..(j-1)]. As a result, dp[i][j] = dp[i][j-1] + d[i-1][j-1]'''
def numDistinct(A,B):
	n = len(A)
	m = len(B)
	if n < m:
		return 0
	dp = [[1] + [0 for j in range(m)] for i in range(n+1)]
	for i in range(n+1):
		for j in range(m+1):
			dp[i][j] = dp[i][j-1]
			if A[i-1] == B[j-1]:
				dp[i][j] = dp[i][j-1] + dp[i-1][j-1]
	return dp[-1][-1]
'''
Given A, B, C, find whether C is formed by the interleaving of A and B.

Input Format:*

The first argument of input contains a string, A.
The second argument of input contains a string, B.
The third argument of input contains a string, C.
Output Format:

Return an integer, 0 or 1:
    => 0 : False
    => 1 : True
Soln Approach:
Given the string S1, S2, S3, the first character of S3 has to match with either the first character of S1 or S2. If it matches with
 first character of S1, we try to see if solution is possible with remaining part of S1, all of S2, and remaining part of S3. Then we do the same thing for S2.

The pseudocode might look something like this :

    bool isInterleave(int index1, int index2, int index3) {
                    // HANDLE BASE CASES HERE
        
        bool answer = false; 
        if (index1 < s1.length() && s1[index1] == s3[index3]) answer |= isInterleave(index1 + 1, index2, index3 + 1);
        if (index2 < s2.length() && s2[index2] == s3[index3]) answer |= isInterleave(index1, index2 + 1, index3 + 1);
        
        return answer;
    }'''
def Interleave(A,B,C):
	def helper(A,B,C,cache):
		if len(A) == 0 and len(B) == 0 and len(C) == 0:
			return True
		key = (len(A),len(B),len(C))
		if key in cache:
			return cache[key]
		r = False
		if len(A) > 0 and len(C) > 0 and A[0] == C[0]:
			r = r | helper(A[1:],B,C[1:],cache)
		if len(B) > 0 and len(C) > 0 and B[0] == C[0]:
			r = r | helper(A,B[1:],C[1:],cache)
		cache[key] = r
		return r
	if helper(A,B,C,{}):
		return 1
	return 0
# Scramble String is a hard problem, need to come back to it
'''
Given an 1D integer array A of length N, find the length of longest subsequence which is first increasing then decreasing.
Example Input
Input 1:
 A = [1, 2, 1]
Input 2:
 A = [1, 11, 2, 10, 4, 5, 2, 1]
Example Output
Output 1:
 3
Output 2:
 6
The problem can be solved as follows:
Construct array inc[i] where inc[i] stores Longest Increasing subsequence ending with A[i]. This can be done simply with O(n^2) DP.
Construct array dec[i] where dec[i] stores Longest Decreasing subsequence ending with A[i]. This can be done simply with O(n^2) DP.
Now we need to find the maximum value of (inc[i] + dec[i] - 1)'''
def makedp(arr):# this func finds and updates the len of longeest incre subseq ending with A[i] and stores that in dp[i]
	n = len(arr)
	dp = [0 for i in range(n)]# initialize
	update = [arr[0]]# this is used to take some space while calculating dp[i]
	dp[0] = 1# initialize
	for i in range(1,n):
		if arr[i] > update[-1]:# found the largest elem till now
			update.append(arr[i])
			dp[i] = len(update)
		else:
			indx = bisect_left(update,arr[i])# not the largest elem, so find where it fits in the seq seen so far
			update[indx] = arr[i]# insert in the correct place
			dp[i] = indx + 1# the right update to length
	return dp
def long_incr_decr_subseq(A):
	if len(A) <= 1:
		return A
	increasing_dp = makedp(A)# thisis the inc[i]
	decreasing_dp = makedp(A[::-1])# this is the dec[i]
	m = max(max(increasing_dp),max(decreasing_dp))
	n = len(A)
	rev = decreasing_dp[::-1]# another counter j going from n-1 to 0 over decreasing dp is not reqd when it is reversed
	for i in range(n):
		if increasing_dp[i] + rev[i] - 1 > m:
			m = increasing_dp[i] + rev[i] - 1
	return m
'''
GIven three prime numbers A, B and C and an integer D.
You need to find the first(smallest) D integers which only have A, B, C or a combination of them as their prime factors
Soln Approach: We use the fact that there are only 3 possibilities of getting to a new number : Multiply by A or B or C.

For each of A, B and C, we maintain the minimum number in our set which has not been multiplied with the corresponding prime number yet.
So, at a time we will have 3 numbers to compare.
The corresponding approach would look something like the following :


initialSet = [A, B, C] 
indexInFinalSet = [0, 0, 0]

for i = 1 to D
  M = get min from initialSet. 
  add M to the finalAnswer if last element in finalAnswer != M
  if M corresponds to A ( or in other words M = initialSet[0] )
    initialSet[0] = finalAnswer[indexInFinalSet[0]] * A
    indexInFinalSet[0] += 1
  else if M corresponds to B ( or in other words M = initialSet[1] )
    initialSet[1] = finalAnswer[indexInFinalSet[1]] * B
    indexInFinalSet[1] += 1
  else 
    # Similar steps for C. 
end
for solve(2,3,5,10) the follwoing is seen
{2: 1, 3: 1, 5: 1}
2
{3: 1, 5: 1, 4: 1, 6: 1, 10: 1}
3
{5: 1, 4: 1, 6: 1, 10: 1, 9: 1, 15: 1}
4
{5: 1, 6: 1, 10: 1, 9: 1, 15: 1, 8: 1, 12: 1, 20: 1}
5
{6: 1, 10: 1, 9: 1, 15: 1, 8: 1, 12: 1, 20: 1, 25: 1}
6
{10: 1, 9: 1, 15: 1, 8: 1, 12: 1, 20: 1, 25: 1, 18: 1, 30: 1}
8
{10: 1, 9: 1, 15: 1, 12: 1, 20: 1, 25: 1, 18: 1, 30: 1, 16: 1, 24: 1, 40: 1}
9
{10: 1, 15: 1, 12: 1, 20: 1, 25: 1, 18: 1, 30: 1, 16: 1, 24: 1, 40: 1, 27: 1, 45: 1}
10
{15: 1, 12: 1, 20: 1, 25: 1, 18: 1, 30: 1, 16: 1, 24: 1, 40: 1, 27: 1, 45: 1, 50: 1}
12
{15: 1, 20: 1, 25: 1, 18: 1, 30: 1, 16: 1, 24: 1, 40: 1, 27: 1, 45: 1, 50: 1, 36: 1, 60: 1}
15
{20: 1, 25: 1, 18: 1, 30: 1, 16: 1, 24: 1, 40: 1, 27: 1, 45: 1, 50: 1, 36: 1, 60: 1, 75: 1}
[2, 3, 4, 5, 6, 8, 9, 10, 12, 15]'''
def nextnumber(A,B,C,D):
	dic = {}
	ans = []
	dic[A] = dic[B] = dic[C] = 1
	for i in range(D):
		val = min(dic)# min gives the min value among all the keys, so the numbers are stored as keys and val as 1 i.e. one number is there
		ans.append(val)
		del dic[val]
		dic[val*A] = dic[val*B] = dic[val*C] = 1
	return ans
'''
A message containing letters from A-Z is being encoded to numbers using the following mapping:

 'A' -> 1
 'B' -> 2
 ...
 'Z' -> 26
Given an encoded message A containing digits, determine the total number of ways to decode it.
Soln Approach: It only makes sense to look at 1 digit or 2 digit pairs ( as 3 digit sequence will be greater than 26 ).

So, when looking at the start of the string, we can either form a one digit code, and then look at the ways of forming the rest of
the string of length L - 1, or we can form 2 digit code if its valid and add up the ways of decoding rest of the string of length L - 2.'''
def numdecodings(A):# A is a string
	if len(A) == 0 or A[0] = "0":# No decoding starting with 0 will be a valid decoding. 
		return 0
	dp = [0]*(len(A)+1)# Mark everything as zero initially
	dp [0] = 1# Now that we know that the string does not begin with zero, 
        # the minimum number of decodings for a length 2 string will be 1. 
        # So mark both as 1.
	for i in range(len(dp)):# At every step, we can either decode 1 or 2 characters. Fish them out.
		if A[i-1] != "0":# If we get a valid single number decoding, the number of decodings will
                # same as previous. Because a single valid decoding won't add to your count.
			dp[i] = dp[i-1]
		if "10" <= A[i-1:i+1] <= "26" and i != 1:# Check if a double number decoding is valid. 
                # If it is valid, we need to add everything before this two digit number to the current number.
			dp[i] += dp[i-1]
	return dp[-1]# dp[x] denotes the num of decoding of string with length x
'''
Given an array of non-negative integers, A, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Determine if you are able to reach the last index.'''
def canJump(a):# a is the given arr
        
        # This variable denotes the maximum array elem we can jump
        # initially it is zero
        max_jump = 0
        for i in range(len(a)):
            # If this index not reachable than return 0
            if i > max_jump:# when we are making the ith iter, it means it is possible to reach the ith index till now
                return 0# if not possible, this check will return false/0
            #update max jump
            max_jump = max(max_jump, i + a[i])# stores the farthest jump possible from any index till now
        return 1
'''
Given an array of non-negative integers, A, of length N, you are initially positioned at the first index of the array.
Each element in the array represents your maximum jump length at that position.
Return the minimum number of jumps required to reach the last index.
If it is not possible to reach the last index, return -1.'''

# Returns minimum number of jumps to reach arr[n-1] from arr[0] 
def minJumps(arr, n): 
  # The number of jumps needed to reach the starting index is 0 
  if (n <= 1): 
    return 0
   
  # Return -1 if not possible to jump 
  if (arr[0] == 0): 
    return -1
   
  # initialization 
  # stores all time the maximal reachable index in the array 
  maxReach = arr[0]   
  # stores the amount of steps we can still take 
  step = arr[0] 
  # stores the amount of jumps necessary to reach that maximal reachable position 
  jump = 1 
   
  # Start traversing array 
   
  for i in range(1, n): 
    # Check if we have reached the end of the array 
    if (i == n-1): 
      return jump 
   
    # updating maxReach 
    maxReach = max(maxReach, i + arr[i]) 
   
    # we use a step to get to the current index 
    step -= 1; 
   
    # If no further steps left 
    if (step == 0): 
      # we must have used a jump 
      jump += 1
        
      # Check if the current index / position or lesser index 
      # is the maximum reach point from the previous indexes 
      if(i >= maxReach): 
        return -1
   
      # re-initialize the steps to the amount 
      # of steps to reach maxReach from position i. 
      step = maxReach - i; 
  return -1

#Another shorter soln maybe
def jump(A):
        
        last = len(A) - 1
        jumps = 0
        reachable = 0      # reachable with current number of jumps 
        next_reachable = 0 # reachable with one additionnal jump
        for i, x in enumerate(A):
            if reachable >= last:# already reched the end
                break 
            if reachable < i:
                reachable = next_reachable
                jumps += 1# by defn of next_reachable
                if reachable < i:# even after the jump not possible means we cannot reach there
                    return -1
            next_reachable = max(next_reachable, i+x)# this is the i + Arr[i] step
        return jumps
'''
Given two integer arrays A and B of size N each which represent values and weights associated with N items respectively.
Also given an integer C which represents knapsack capacity.
Find out the maximum value subset of A such that sum of the weights of this subset is smaller than or equal to C.
Soln Approach: There can be two cases for every item:

the item is included in the optimal subset.
not included in the optimal set.
Therefore, the maximum value that can be obtained from n items is max of following two values.

Maximum value obtained by n-1 items and W weight (excluding nth item).
Value of nth item plus maximum value obtained by n-1 items and W minus weight of the nth item (including nth item).
If weight of nth item is greater than W, then the nth item cannot be included and case 1 is the only possibility.

We will solve it by using DP with the bottom-up approach. Our knapsack size is W, we have to make maximum value to fill the knapsack.
 A simple approach will be, how can we get maximum value if your knapsack size 1, then compute maximum value if knapsack size is 2 and
  so on….

Suppose dp[i][j] represents the maximum value that can be obtain considering first i items and a knapsack with a capacity of j.
Then our recurrence relation will look like:
dp[i][j]=max(dp[i-1][j] (When we don’t consider this item) or dp[i-1][j-wt[i]]+val[i] (When we consider this item) )

Time Complexity: O(NW)'''
def knapsack(capacity,weights,values):
	n = len(values)
	dp = [[0 for x in range(capacity+1)] for y in range(n+1)]
	for i in range(n+1):
		for j in range(capacity+1):
			if i == 0 or j == 0:
				dp[i][j] = 0
			if weights[i-1] < capacity:
				dp[i][j] = max(dp[i-1][j],dp[i-1][j - weights[i]] + values[i-1])
			else:
				dp[i][j] = dp[i-1][j]
	return dp[n][capacity]
# Equal Average Partition is a hard problem to be solved later
'''
Given the stock price of n days, the trader is allowed to make at most k transactions, where a new transaction can only start after the
 previous transaction is complete, find out the maximum profit that a share trader could have made.
Input:  
Price = [10, 22, 5, 75, 65, 80]
    K = 2
Output:  87
Trader earns 87 as sum of 12 and 75
Buy at price 10, sell at 22, buy at 
5 and sell at 80
Let profit[t][i] represent maximum profit using at most t transactions up to day i (including day i). Then the relation is:

profit[t][i] = max(profit[t][i-1], max(price[i] – price[j] + profit[t-1][j]))
          for all j in range [0, i-1]
profit[t][i] will be maximum of –

profit[t][i-1] which represents not doing any transaction on the ith day.
Maximum profit gained by selling on ith day. In order to sell shares on ith day, we need to purchase it on any one of [0, i – 1] days.
 If we buy shares on jth day and sell it on ith day, max profit will be price[i] – price[j] + profit[t-1][j] where j varies from 0 to i-1.
 Here profit[t-1][j] is best we could have done with one less transaction till jth day.
The above solution has time complexity of O(k.n2). It can be reduced if we are able to calculate the maximum profit gained by selling
 shares on the ith day in constant time.

profit[t][i] = max(profit [t][i-1], max(price[i] – price[j] + profit[t-1][j]))
                            for all j in range [0, i-1]

If we carefully notice,
max(price[i] – price[j] + profit[t-1][j])
for all j in range [0, i-1]

can be rewritten as,
= price[i] + max(profit[t-1][j] – price[j])
for all j in range [0, i-1]
= price[i] + max(prevDiff, profit[t-1][i-1] – price[i-1])
where prevDiff is max(profit[t-1][j] – price[j])
for all j in range [0, i-2]

So, if we have already calculated max(profit[t-1][j] – price[j]) for all j in range [0, i-2], we can calculate it for j = i – 1 in
 constant time. In other words, we don’t have to look back in the range [0, i-1] anymore to find out best day to buy. We can determine
  that in constant time using below revised relation.

profit[t][i] = max(profit[t][i-1], price[i] + max(prevDiff, profit [t-1][i-1] – price[i-1])
where prevDiff is max(profit[t-1][j] – price[j]) for all j in range [0, i-2]'''
def maxProfit(price,k):# k is the maximum number of transactions allowed
  
    # Table to store results of subproblems  
    # profit[t][i] stores maximum profit  
    # using atmost t transactions up to  
    # day i (including day i)
    n = len(price)# price is the arr given with the price[i] being the val of the stock in ith day
    profit = [[0 for i in range(n + 1)]  
                 for j in range(k + 1)]  
  
    # Fill the table in bottom-up fashion  
    for i in range(1, k + 1):  
        prevDiff = float('-inf') 
          
        for j in range(1, n):  
            prevDiff = max(prevDiff, profit[i - 1][j - 1] - price[j - 1])  
            profit[i][j] = max(profit[i][j - 1],  price[j] + prevDiff) 
    return profit[k][n - 1]
#The time complexity of the above solution is O(kn) and space complexity is O(nk)
# if infinite transactions are allowed, then for the soln approach look at the following link
#https://www.youtube.com/watch?v=HWJ9kIPpzXs
def maxProfit(self, A):
        d = 0
        n = len(A)
        for i in range(1,n):
            if A[i]>A[i-1]:# this finds exactly the peaks as reqd
                d += A[i] - A[i-1]# if the selling price grows up, then the sum adds and cancels previous values in telescoping fashion
        return d
# Shortest common superstring is a hard problem to be solved later
'''
Given a string A containing just the characters ’(‘ and ’)’.
Find the length of the longest valid (well-formed) parentheses substring
Soln approach: Lets construct longest[i] where longest[i] denotes the longest set of parenthesis ending at index i.

If s[i] is ‘(‘, set longest[i] to 0, because any string end with ‘(‘ cannot be a valid one.
Else if s[i] is ‘)’
If s[i-1] is ‘(‘, longest[i] = longest[i-2] + 2
Else if s[i-1] is ‘)’ and s[i-longest[i-1]-1] == ‘(‘, longest[i] = longest[i-1] + 2 + longest[i-longest[i-1]-2]'''
def longestValidParen(s):
	n = len(s)
	longest = [0]*n
	maxlen = 0
	for i in range(n):
		if s[i] == ")":
			if i-1 >= 0 and s[i-1] == "(":
				longest[i] = longest[i-1] + 2
				maxlen = max(maxlen,longest[i])
			else:
				if i-longest[i-2]-1 >= 0 and longest[i-longest[i-2]-1] == "(":
					longest[i] = longest[i-2] + 2 + (longest[i-longest[i-2]-2] if i-longest[i-2]-2 >= 0 else 0)
					maxlen = max(maxlen,longest[i])
	return maxlen
'''
Given a 2D integer matrix A of size N x M.
From A[i][j] you can move to A[i+1][j], if A[i+1][j] > A[i][j], or can move to A[i][j+1] if A[i][j+1] > A[i][j].
The task is to find and output the longest path length if we start from (0, 0)
Soln approach: Maintain the 2D matrix, dp[][], where dp[i][j] store the value of length of longest increasing sequence for sub matrix
 ending from ith row and j-th column.

Recurrence Relation looks like:

If(A[i][j] > A[i][j-1])
dp[i][j] = max(dp[i][j], dp[i][j-1] + 1)
If(A[i][j] > A[i-1][j])
dp[i][j] = max(dp[i][j], dp[i-1][j] + 1)

Time Complexity: O(NM)
Space Complexity: O(NM)'''
def longestpath(A):
	dp = [[-1 for i in range(len(A[0]))] for j in range(len(A))]
	for i in range(len(A)):
		for j in range(len(A[0])):
			if i == j == 0:
				dp[i][j] = 1
			if j!= 0 and A[i][j] > A[i][j-1] and dp[i][j-1] != -1:
				dp[i][j] = dp[i][j-1] + 1
			if i!= 0 and A[i][j] > A[i-1][j] and dp[i-1][j] != -1:
				dp[i][j] = dp[i][j-1] + 1
	return dp[-1][-1]
# now if we want to solve the same problem in less than O(n^2) time, here is th soln
'''
Lets max_x[i][j] denote the length of 1s in the same row i starting from (i,j).

So our current max with one end of the rectangle at (i,j) would be max_x[i][j].
As we move to the next row, there are 2 cases :
1) max_x[i+1][j] >= max_x[i][j] which means that we can take max_x[i][j] 1s from next column as well and extend our current rectangle
 as it is, with one more extra row.
11100000 - 111
11111100 - 111

2) max_x[i+1][j] < max_x[i][j] which means that if we want to extend our current rectangle to next row, we need to reduce the number
 of columns in it to max_x[i+1][j]
11100000 - 11
11000000 - 11

As mentioned above, we keep increasing the columns and adjusting the width of the rectangle.
O(N^3) time complexity.

Even though N^3 is acceptable, it might be worth exploring a better solution.
If you notice, laying out max_x[i][j] helps you make histograms in every row. Then the problem becomes of finding the maximum area
 in histograms ( which we have solved before in Stacks and Queues ) in O(n). This would lead to an O(N^2) solution. '''
def maximalRectangle(matrix):
        heights = [0]*(len(matrix[0])+1) # why +1 here
        max_area = 0
        for row in matrix:
            for i in range(len(matrix[0])):
                heights[i] = heights[i]+1 if row[i] else 0
            stack = []
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i - stack[-1] - 1 if stack else i
                    max_area = max(max_area, h*w)
                stack.append(i)
        return max_area
# Given a 2D binary matrix A of size  N x M  find the area of maximum size square sub-matrix with all 1's.
def solve(A):
        """ dynamic programming approach """
        if not A:
            return 0
        row = len(A) + 1
        col = len(A[0]) + 1
        # row and col are incremented by 1 to provide padding of zeroes along rows and columns
        dp = [[0]*col for _ in range(row)]
        
        maxDimension = 0 # longest possible length of side of the square
        for i in range(1, row):
            for j in range(1, col):
                if A[i - 1][j - 1] :
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                else:
                    dp[i][j] = 0
                maxDimension = max(maxDimension, dp[i][j])
                    
        return maxDimension * maxDimension
'''
Given a grid of size m * n, lets assume you are starting at (1,1) and your goal is to reach (m,n). At any instance, if you are on (x,y), 
you can either go to (x, y + 1) or (x + 1, y).
Now consider if some obstacles are added to the grids. How many unique paths would there be?
An obstacle and empty space is marked as 1 and 0 respectively in the grid.'''
def uniquePathsWithObstacles(A, p = (0, 0)):
        i , j = p
        if i >= len(A) or j >= len(A[0]):# already reached
            return 0
            
        if A[i][j] == 1:# cant move at all due to obstacles
            return 0
        if i == len(A) - 1 and j == len(A[0]) - 1:# just one step away
            return 1
        right = uniquePathsWithObstacles(A, (i , j + 1))# recursive calls    
        down = uniquePathsWithObstacles(A, (i + 1, j))
        return right + down
'''
Given a string A and a dictionary of words B, determine if A can be segmented into a space-separated sequence of one or more
 dictionary words.
You start exploring every substring from the start of the string, and check if its in the dictionary. 
If it is, then you check if it is possible to form rest of the string using the dictionary words. If yes, my answer is true.
 If none of the substrings qualify, then our answer is false.'''
def wordbreak(word,arr):
	dp = [0]*(len(word) +1)# the dp stores 1 and 0. 1 is stored at index i if we have matched A[:i+1] with some word or other from arr
	dp[0] = 1# this is initialized so that we can carry on the dp as in line 629
	for i in range(len(word)):# start matching from the beginning of the word
		for w in arr:
			if word[i+1-len(w):i+1] == w and dp[i+1-len(w):i+1] == 1:# w fits inside word and where it fits in, the word is already matched till the last index
				dp[i+1] = 1# now it gets matched to the i+1 index
				break# we leave this loop of searching for words in arr, go to the next index and search for words in arr again
	return dp[-1]# the last index if 1 means we have matched word till the last index incrementally
'''
Given an integer A, how many structurally unique BST’s (binary search trees) exist that can store values 1…A?
Lets say you know the answer for values i which ranges from 0 <= i <= n - 1.
How do you calculate the answer for n.

Lets consider the number [1, n]
We have n options of choosing the root.
If we choose the number j as the root, j - 1 numbers fall in the left subtree, n - j numbers fall in the right subtree. We already know how many ways there are to forming j - 1 trees using j - 1 numbers and n -j numbers.
So we add number(j - 1) * number(n - j) to our solution.'''
def numTrees(self, A):
        dp=[0]*(A+1)
        if(A==1):return 1
        dp[0]=1;dp[1]=1
        for i in range(2,A+1):
            for j in range(i):
                dp[i]+=dp[j]*dp[i-j-1]
        return dp[-1]
'''
Given a N * 2 array A where (A[i][0], A[i][1]) represents the ith pair.
In every pair, the first number is always smaller than the second number.
A pair (c, d) can follow another pair (a, b) if b < c , similarly in this way a chain of pairs can be formed.
Find the length of the longest chain subsequence which can be formed from a given set of pairs.
Soln approach: This problem is a variation of standard Longest Increasing Subsequence problem. Following is a simple two step process.

Run a modified LIS process where we compare the second element of already finalized LIS with the first element of new LIS being constructed.'''
def longestchain(A):
	dp = [1]*len(A)
	for i in range(len(A)):
		for j in range(i):
			if A[i][0] > A[j][1] and dp[j] + 1 < dp[i]:
				dp[i] = dp[j] + 1
	return dp[-1] 
'''
Given a 2 x N grid of integer, A, choose numbers such that the sum of the numbers
is maximum and no two chosen numbers are adjacent horizontally, vertically or diagonally, and return it.
V : 
1 |  2  |  3  | 4
2 |  3  |  4  | 5

Lets first try to reduce it into a simpler problem. 
We know that within a column, we can choose at max 1 element. 
And choosing either of those elements is going to rule out choosing anything from the previous or next column. 
This means that choosing V[0][i] or V[1][i] has identical bearing on the elements which are ruled out. 
So, instead we replace each column with a single element which is the max of V[0][i], V[1][i].

Now we have the list as : 
2 3 4 5

Here we can see that we have reduced our problem into another simpler problem.
Now we want to find maximum sum of values where no 2 values are adjacent. 
Now our recurrence relation will depend only on position i and,
 a "include_current_element" which will denote whether we picked last element or not.
  
MAX_SUM(pos, include_current_element) = 
IF include_current_element = FALSE THEN   
	max | MAX_SUM(pos - 1, FALSE) 
	    | 
	    | MAX_SUM(pos - 1, TRUE)

ELSE    |
	MAX_SUM(pos - 1, FALSE) + val(pos)'''
def adjacent( A):
	        '''let m, n be the best solutions obtained so far by respectively picking:
	           m: penulitimate element
	           n: last element
	        Time complexity: O(n),  constant space.
	    '''
	    m, n = 0, 0
	    for a, b in zip(*A):# this forms tuples taking each column of A
	        x = max(a, b)# line 699-700 says about this, we take only the largest elem
	        # update m (not picking x) and n (picking x)
	        m, n = max(m, n), m+x# -----------m--n--x is the situation, max(m,n) becomes the optimal soln if we do not pick x, else we do not pick n
	    return max(m, n)
'''
You are given a set of coins S. In how many ways can you make sum N assuming you have infinite amount of each coin in the set.
Note : Coins in set S will be unique. Expected space complexity of this problem is O(N).
Lets say we can make the sum N - S[i] in X ways. Then if we have a coin of value S[i] we can also make a sum of N in X ways.
 We can memoize the number of ways in which we can make all the sums < N. This can be done by keeping a count array for all sums
  less than N which gives us the expected space complexity of O(N). A sum of 0 is always possible as we can pick no coins, so the 
  base case will be count[0] = 1'''
def numberofways(coins,sum):
	dp = [0]*(sum+1)
	for coin in coins:
		for i in range(1,sum+1):
			if coin <= i:
				dp[i] += dp[i-coin]
	return dp[-1]
'''
Find the contiguous subarray within an array (containing at least one number) which has the largest product.
Return an integer corresponding to the maximum product possible

If there were no zeros or negative numbers, then the answer would definitely be the product of the whole array.
Now lets assume there were no negative numbers and just positive numbers and 0. In that case we could maintain a current maximum product
 which would be reset to A[i] when 0s were encountered.
When the negative numbers are introduced, the situation changes ever so slightly. We need to now maintain the maximum product in positive
 and maximum product in negative. On encountering a negative number, the maximum product in negative can quickly come into picture.'''
def maxProduct(A):
        assert len(A) > 0
        ans = A[0]
        ma, mi = 1, 1
        for a in A:
            ma, mi = max(a, a*ma, a*mi), min(a, a*ma, a*mi)# when we find a neg num, then ma becomes the next a after that neg num
            ans = max(ans, ma, mi)
        return ans
'''
There are a row of N houses, each house can be painted with one of the three colors: red, blue or green.
The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.
The cost of painting each house with a certain color is represented by a N x 3 cost matrix A.
For example, A[0][0] is the cost of painting house 0 with color red; A[1][2] is the cost of painting house 1 with color green, and so on.
Find the minimum total cost to paint all houses

Let:

dp[i][0] represent the minimum total cost to paint the houses till i where house i is colored in red.
dp[i][1] represent the minimum total cost to paint the houses till i where house i is colored in green.
dp[i][2] represent the minimum total cost to paint the houses till i where house i is colored in blue.
So if you paint house ‘i’ with red then you can paint house ‘i-1’ only in blue or green.
So dp[i][0] = A[i][0] + min(dp[i-1][1],dp[i-1][2])
Similarly:
dp[i][1] = A[i][1] + min(dp[i-1][0], dp[i-1][2])
dp[i][2] = A[i][2] + min(dp[i-1][0], dp[i-1][1])

At last output the minimum of (dp[n-1][0],dp[n-1][1],dp[n-2][2)

Time Complexity:O(N).
Space Complexity:O(N).

Bonus:Try to think of a constant space solution.'''
def solve(arr):
        n = len(arr)
        # DP STATE
        # dp<Color>[i] = Minimum Cost of painting till ith House with 'Color'
        # Base Conditions = dp<Color>[0] = 0
        dpR = [0] * (n+1)
        dpG = [0] * (n+1)
        dpB = [0] * (n+1)
        #DP EXPRESSION
        # dp<Color>[i] = min(dp<other_colors>[i-1]) + Cost of this Color i.e arr[i][<color>]
        for i in range(1,n+1):
            dpR[i] = min(dpG[i-1], dpB[i-1]) + arr[i-1][0]
            dpG[i] = min(dpR[i-1], dpB[i-1]) + arr[i-1][1]
            dpB[i] = min(dpR[i-1], dpG[i-1]) + arr[i-1][2]
    
        #print(f'dpR: {dpR}')
        #print(f'dpG: {dpG}')
        #print(f'dpB: {dpB}')
    
        # Answer at min(dp<Color>[N]) - Minimum of all states to color till Nth Houses.
        return min(dpR[n], dpG[n], dpB[n])
'''
Given a string A, partition A such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of A.'''
def minCut(s):
        if len(s) == 0: return 0
        n = len(s)
        dp = [[False for x in range(n)] for x in range(n)]
        d = [0 for x in range(n)]
        for i in range(n-1,-1,-1):
            d[i] = n-i-1
            for j in range(i,n):
                if s[i] == s[j] and (j-i < 2 or dp[i+1][j-1]):
                    dp[i][j] = True
                    if j == n-1:
                        d[i] = 0
                    elif d[j+1] + 1 < d[i]:
                        d[i] = d[j+1] + 1
        return d[0]
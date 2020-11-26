import math
import sys
from collections import Counter
from collections import defaultdict

# Given an array containing list of numbers, find all subsets of that set of numbers and output as list of lists
def subsets(s):
	r = [[]]
	for e in s:
		#print([x+[e] for x in r])
		r += [x + [e] for x in r]
		#print(r)
	return r
# id we print the commented lines for s = [1,2,3], look at the output, the algorithm will be clear
'''[[1]]
[[], [1]]
[[2], [1, 2]]
[[], [1], [2], [1, 2]]
[[3], [1, 3], [2, 3], [1, 2, 3]]
[[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]'''

# Given a list A of numbers [1,2,3...,A] and a number B, return all  possible combinations of B numbers out of those A numbers, so
# that the combinations are sorted

def comb(A,B):
	if B <=0:
		return []
	if B == 1:
		return [[a] for a in A]
	ans = []
	for i in range(len(A)):
		ans += [[A[i]] + c for c in comb(A[i+1:], B-1)]
	return ans
# Now if we have written the following after the for loop and run the func for comb([1,2,3,4],2):
''' for i in range(len(A)):
		print(comb(A[i+1:],B-1))
		print([A[i]])
		print([[A[i]]+c for c in comb(A[i+1:],B-1)])
		print(ans)
		ans += [[A[i]]+c for c in comb(A[i+1:],B-1)]
		print(ans)

Then the answer will look like the following:

[[2], [3], [4]]
[1]
[[1, 2], [1, 3], [1, 4]]
[]
[[1, 2], [1, 3], [1, 4]]
[[3], [4]]
[2]
[[2, 3], [2, 4]]
[[1, 2], [1, 3], [1, 4]]
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4]]
[[4]]
[3]
[[3, 4]]
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4]]
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
[]
[4]
[]
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
[[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

The logic is that we are picking the ith element out A[i] and then picking B-1 elem out of the elements A[i+1:] and then inserting A[i]
in the result to make it group of B. we are doing this for every i, so even in A[i+1:] when we are missing elems from the front
we have already counted when i was small. B-1 elems out of A[i+1:] is done recursivley and the new results formed are added to ans
which is a list of lists.'''

#Given a collection of numbers in a list A, return all possible permutations, as a list of lists

def permute(A):
	if len(A) == 1:
		return [[A]]
	result = []
	for i in range(len(A)):# each of the elems in A should be in the first position for many permutations
		nxt = permute(A[:i] + A[i+1:])# we pick out the first elem and generate all permutations that should have A[i] at its beginning
		for j in nxt:# for each such perm
			result.append([A[i]] +j)# create the reqd perm with A[i] in the start
	return result

'''
Given an integer n, return all distinct solutions to the n-queens puzzle.Each solution contains a distinct board configuration of
 the n-queens’ placement, where 'Q' and '.' both indicate a queen and an empty space respectively.
There exist two distinct solutions to the 4-queens puzzle:
[
 [".Q..",  // Solution 1
  "...Q",
  "Q...",
  "..Q."],

 ["..Q.",  // Solution 2
  "Q...",
  "...Q",
  ".Q.."]
]

The algo we will follow is the following:
1) Start in the toptmost row
2) If all queens are placed
    return true
3) Try all cols in the current row.  Do following
   for every tried col.
    a) If the queen can be placed safely in this col
       then mark this [row, column] as part of the 
       solution and recursively check if placing  
       queen here leads to a solution.
    b) If placing queen in [row, column] leads to a
       solution then return true.
    c) If placing queen doesn't lead to a solution 
       then unmark this [row, column] (Backtrack) 
       and go to step (a) to try other cols.
3) If all cols have been tried and nothing worked, 
   return false to trigger backtracking.
'''
def solveNqueens(A):
	board = [["."]*A for _ in range(A)]# creating the board with no queens

	def check(row,col,board):
		for i in range(A):
			for j in range(A):
				if board[i][j] == "Q":# if we find a Q, then check for the rows/cols/diag/offdiag which contains thatQ
					if i == row or j== col or i+j == row + col or i-j == row - col:# checks if there is a Q in the same row/col/diag/off-daig
						return False
		return True

	def solve_util(board,row,ans):#It mainly solve the problem. It returns false if queens cannot be placed, otherwise return true and 
								#prints placement of queens 
		if row == A:#base case: If all queens are placed then return true. We are putting Q starting from topmost row anf this means the
		# last row has been reached, so we have got a soln, else, we could not have reached last row, we would have got a False even before that
			to_append = ["".join(s) for s in board]# this is where we are printing the soln in the reqd format
			ans.append(to_append)# we are returning this ans at the end
			return
		for c in range(A):# for a particular row, we check all the cols
			if check(row,c,board):# this is the function which returns T/F and decides whether to recurse/backtrack
				board[row][c] = "Q"# Place this queen in board[row][c]
				solve_util(board,row+1,ans)# recurse by solving for the next row, Make result true if any placement is possible
			board[row][c] = "."# if the above line returns false at some point in the recursion, we come to this line which backtracks
				#If placing queen in board[row][c] doesn't lead to a solution, then remove queen from board[row][c] and backtrack
	ans = list()
	solve_util(board,0,ans)# we start from the top row=0 and recurse, list is empty	
	return ans

#Given a digit string, return all possible letter combinations that the number could represent.
#{'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz', '0': '0', '1': '1'}
#Input: Digit string "23"
#Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]

def lettercomb(digits):
	mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz', '0': '0', '1': '1'}
	if len(digits) == 0:
		return []
	if len(digits) == 1:
		return list(mapping[digits[0]])
	prev = lettercomb(digits[:-1])
	later = mapping[digits[-1]]
	return [s+c for s in prev and c in later]

#Given a collection of integers that might contain duplicates, S, return all possible subsets
'''
Elements in a subset must be in non-descending order.
The solution set must not contain duplicate subsets.
The subsets must be sorted lexicographically.
If S = [1,2,2], the solution is:

[
[],
[1],
[1,2],
[1,2,2],
[2],
[2, 2]
]
'''
def subsetsdup(A):
	if len(A) == 0:# base case
		return [[]]
	A.sort()# original list might not be sorted
	ans = [[]]
	for i in range(A):
		for j in range(len(ans)):
			r = ans[j] + [A[i]]# elements are kept in the subset in non decreasing order by this way
			if r in ans:# checking for duplicates
				continue# skipping that r
			ans.append(r)
	ans.sort()# subsets are sorted lexicographically here
	return ans

'''
Given a string s, partition s such that every string of the partition is a palindrome.

Return all possible palindrome partitioning of s.

For example, given s = "aab",
Return

  [
    ["a","a","b"]
    ["aa","b"],
  ]
  In the given example,
["a", "a", "b"] comes before ["aa", "b"] because len("a") < len("aa")
  '''

def partition(A):
	n = len(A)
	result = []

	def recursive(ans_till_now,starting_index):
		if starting_index == n:# if we have n this means we have all the palindromes needed, the starting index has checked
		# for len n-1 and has now been incremented and so the final ans is a list which to be added in the list result
			result.append(ans_till_now)
			return
		for i in range(starting_index,n):#When on index i, you incrementally check all substring starting from i for being
    # palindromic. If found, you recursively solve the problem for the remaining string
    # and add it to your solution. Start this recursion from starting position of the
    # string. the i is actually the length of palindromes, first len 1 palindromes, then len 2 palindromes, etc. always start to look
    # from the beginning of string. if we are looking for x len palindromes and the first x len part of string is not palindrome
    # this means the str cannot be decomposed into this part, start with len x+1
			temp = A[starting_index:i+1]
			if temp == temp[::-1]:# if palindrome, add to the ans till now
				recursive(ans_till_now+[temp],i+1)# recursively search for next
	recursive([],0)
	return result# list of lists

#Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses of length 2*n
'''
Approach: To form all the sequences of balanced bracket subsequences with n pairs. So there are n opening brackets and n closing brackets.
So the subsequence will be of length 2*n. There is a simple idea, the i’th character can be ‘{‘ if and only if the count of
 ‘{‘ till i’th is less than n and i’th character can be ‘}’ if and only if the count of ‘{‘ is greater than the count of ‘}’ till
  index i. If these two cases are followed then the resulting subsequence will always be balanced.
So form the recursive function using the above two cases.

Algorithm:

Create a recursive function that accepts a string (s), count of opening brackets (o) and count of closing brackets (c) and the value of n.
if the value of opening bracket and closing bracket is equal to n then print the string and return.
If the count of opening bracket is greater than count of closing bracket then call the function recursively with the following
 parameters String s + “}”, count of opening bracket o, count of closing bracket c + 1, and n.
If the count of opening bracket is less than n then call the function recursively with the following parameters
 String s + “{“, count of opening bracket o + 1, count of closing bracket c, and n.
'''
def generateparenthesis(A):
	def generate(open,close,n,str):
		if close == n:
			return [s]
		res = []
		if open < n:
			res +=  generate(open+1,close,n,str + "{")
		if open > close:
			res +=  generate(open,close+1,n, str + "}")
		return res
	return generate(0,0,A,"")

#Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers
# sums to T.
'''
Given candidate set 10,1,2,7,6,1,5 and target 8,

A solution set is:

[1, 7]
[1, 2, 5]
[2, 6]
[1, 1, 6]
All numbers (including target) will be positive integers.
Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
The solution set must not contain duplicate combinations.
'''
def combinationsum(A,B):
	A.sort()
	ans = set()
	def helper(arr,sumSoFar,index):
		if sumSoFar == B:
			ans.add(tuple(sumSoFar))
		if sumSoFar > B:
			return
		else:
			for i in range(index,len(A)):
				helper(arr+[A[i]],sumSoFar+A[i],index+1)
	helper([],0,0)
	return list(ans)

#The set [1,2,3,…,n] contains a total of n! unique permutations. By listing and labeling all of the permutations in order
#Given n and k, return the kth permutation sequence
'''
What if n is greater than 10. How should multiple digit numbers be represented in string?
 In this case, just concatenate the number to the answer.
so if n = 11, k = 1, ans = "1234567891011" 
Whats the maximum value of n and k?
 In this case, k will be a positive integer thats less than INT_MAX.
n is reasonable enough to make sure the answer does not bloat up a lot. 

Approach: Number of permutation possible using n distinct numbers = n!

Lets first make k 0 based.
Let us first look at what our first number should be.
Number of sequences possible with 1 in first position : (n-1)!
Number of sequences possible with 2 in first position : (n-1)!
Number of sequences possible with 3 in first position : (n-1)!

Hence, the number at our first position should be k / (n-1)! + 1 th integer.
'''

def findperm(n,k):
	numbers = list(range(1,n+1))
	perm = ""
	k = k-1
	while(n>0):
		n -= 1
		index,k = divmod(k,factorial(n))# get the index of current digit, index is quotient and k on LHS is the new remainder
		perm += str(numbers[index])# append to ans
		numbers.remove(numbers[index])# this number cannot occur again, all nos distinct in a perm
	return perm

'''
Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
The same repeated number may be chosen from C unlimited number of times
Given candidate set 2,3,6,7 and target 7,
A solution set is:

[2, 2, 3]
[7]
'''
def combinsum(C,target):
	def comb(c,t,prefix = []):
		if not c:# empty
			return
		if t < 0:# backtrack to previous levels
                # the lowest number in C > required target sum
			return
		if t == 0:# target-sum can achieved with prefix+C[0]
                # add to results, and backtrack to previous levels
			result.add(tuple(prefix))
		comb(c,t-c[0],prefix + [c[0]])# this is not backtracking, this is looking for more solutions
		comb(c[1:],t,prefix + [c[0]])# upon backtrack, reduce candidate-set and try with the same prefix/target sum
	C.sort()
	result = set()
	comb(C,target)
	return sorted(map(list,result))
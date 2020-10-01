import math
import sys
import string
from collections import deque

#Given an array, print the Next Greater Element (NGE) for every element. The Next greater Element for an element x is the first greater
# element on the right side of x in array. Elements for which no greater element exist, consider next greater element as -1

#Solution using stack:
"""
So to find next greater element, we used stack one from left and one from right.simply we are checking which element is greater and storing
 their index at specified position. The following algo and the code is for the prev greater elem
1- if stack is empty, push current index.
2- if stack is not empty
a)if current element is greater than top element then store the index of current element on index of top element.
Take the following eg and dryrun the below code on this, everything will be clear
eg. for array [2, 25, 20, 11, 21, 3]
stack A -> emtpy
"""

def prevgreater_elem(arr):# for the nextgreater elem, go from right, i.e i ranges in (len(arr)-1,-1,-1), everything else same
	left_list = [0]* len(arr)# the answer will contain right elems for each elem in arr at the exact indices
	stack = []# contains indices of elems from arr only
	for i in range(len(arr)):# moving back to find the prev
		while (stack != [] and arr[i] >= arr[stack[-1]]):# this is the condition to be satisfied, cant pop if stack = []
			stack.pop()# popping till we find the val in stack just greater than arr[i], so this should be the ans for arr[i].
		if stack != []:# even after popping if something is in stack then it is the match we were looking for
			left_list[i] = stack[-1]# so got the ans
		if stack == []:# this means stack has got empty while popping all elems in the while loop, so nothing in the left anymore
			left_list[i] = 0# so no elem on the left of the current index satisfies the condition
		stack.append(i)# at the end always add the current index at the top of the stack at the end of each round of loop
	return left_list

#Given an array of integers A of size N. A represents a histogram i.e A[i] denotes height of
#the ith histogram’s bar. Width of each bar is 1.
""" Solution: To find the maximal rectangle, if for every bar x, we know the first smaller bar on its each side, let's say l and r,
we are certain that height[x] * (r - l - 1) is the best shot we can get by using height of bar x. OK, let's assume we can do this in
O(1) time for each bar, then we can solve this problem in O(n)! by scanning each bar. use an increasing stack can keep track of the
first smaller on its left and right. for stack[x], stack[x-1] is the first smaller on its left, then a new element that can pop stack[x]
out is the first smaller on its right."""

def maxrect(heights):# heights is the arr of bars with hts
	stack = [-1]# we never have to check if stack is empty if we are popping indices from stack, since -1 will never get popped.
	heights.append(0)# all hts are greater than 0, so the rectangle with smallest bar as height will be considered too when that height
	# ht will be pooped from the stack(actually index of that height since stack contains indices only)
	area = 0
	for i in range(len(heights)):# go over the whole histogram
		while heights[i] < heights[stack[-1]]:# the next elem in arr is smaller than top of stack, so this next elem is the first smaller
		# elem to the right of the elem on top of stack. the first smaller elem to the left of the elem on top of the stack is the elem
		# just below the top of the stack, since we are maintaining the stack in increasing fashion.
			ht = stack.pop()# this is the elem on top of the stack, we want to find best rect with this height
			base = i-stack[-1]-1# i is the current index which gives the index of the bar which is smaller than the top of the stack just
			# to its right. stack[-1] is now the 2nd top elem in the stack which is the smaller elem than the top just to the left of the
			# left of the top. the base of the rect whould be right-left -1. area is ht*(rt-left-1).
			area = max(area,ht*base)# updating the max here
		stack.append(i)# if ht[i] > top of stack, then we are continuing in the increasing fashion, so just append it.
	return ans# note that the bar with the lowest ht will be pooped at the end of all elems in the stack and it will be pooped only because
	# we have appended 0 to the heights arr. We are always trying to find the best rect with the height acc to the top of the stack.

"""Given a string A denoting a stream of lowercase alphabets. You have to make new string B.
B is formed such that we have to find first non-repeating character each time a character is inserted to the stream and append it at
 the end to B. If no non-repeating character is found then append '#' at the end of B.

"abcabc" ---> "aaabc#"
	"a"      -   first non repeating character 'a'
    "ab"     -   first non repeating character 'a'
    "abc"    -   first non repeating character 'a'
    "abca"   -   first non repeating character 'b'
    "abcab"  -   first non repeating character 'c'
    "abcabc" -   no non repeating character so '#'
"""

def solve(A):# A is the str and at every iteration we go through either if or elif
	queue = []# the first elem is the first nonrepeated elem in the str as we store them in that order, if repeated char, remove that char
	visisted = set()# we put any char here only first time we see it, all the chars in the str occur here once
	repeated = set()# 2nd time onwards a char is seen we put it here but only one copy of that char is kept since this is a set
	result = []# this is the result we will finally output
	for a in A:
		if a not in visisted:# we are seeing this char the first time
			queue.append(a)# added to the queue in that order
			visisted.add(a)# so seen 1st time, now add to the set
		elif a not in repeated:# we have come here since it is in visited, so char is seen 2nd/3rd/4th/.. time now
			repeated.add(a)# so add to this set
			queue.remove(a)# remove this since this is no longer the first non repeated char, this is a repeated char
		if queue:
			letter = queue[0]# the first elem in the queue is the first non repeated char always
		else:
			letter = "#"# if queue is empty, so all chars are removed from queue, so all of them have repeated
		result.append(letter)# letter is the first non repeated char after every iteration, so add it to the result
	return "".join(result)# converting to str

#Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able
# to trap after raining.

def findWater(arr, n): 
  
    # left[i] contains height of tallest bar to the 
    # left of i'th bar including itself 
    left = [0]*n 
  
    # Right [i] contains height of tallest bar to 
    # the right of ith bar including itself 
    right = [0]*n 
  
    # Initialize result 
    water = 0
  
    # Fill left array 
    left[0] = arr[0] 
    for i in range( 1, n): 
        left[i] = max(left[i-1], arr[i]) 
  
    # Fill right array 
    right[n-1] = arr[n-1] 
    for i in range(n-2, -1, -1): 
        right[i] = max(right[i + 1], arr[i]); 
  
    # Calculate the accumulated water element by element 
    # consider the amount of water on i'th bar, the 
    # amount of water accumulated on this particular 
    # bar will be equal to min(left[i], right[i]) - arr[i] . 
    for i in range(0, n): 
        water += min(left[i], right[i]) - arr[i] 
  
    return water 

#Given an array and an integer K, find the maximum for each and every contiguous subarray of size k

def printmax(arr,k):
	n = len(arr)
	Q = deque()
	""" Create a Double Ended Queue, Qi that  
    will store indexes of array elements.  
    The queue will store indexes of useful  
    elements in every window and it will 
    maintain decreasing order of values from 
    front to rear in Qi, i.e., arr[Qi.front[]] 
    to arr[Qi.rear()] are sorted in decreasing 
    order"""
	result = []
	for i in range(k):# Process first k (or first window)  
    # elements of array 
		while Q and arr[i] >= arr[Q[-1]]:# For every element, the previous  
        # smaller elements are useless 
        # so remove them from Qi 
			Q.pop()
		Q.append(i)# Add new element at rear of queue 
	for i in range(k,n):# Process rest of the elements, i.e.  
    # from arr[k] to arr[n-1] 
		result.append(Q[0])# The element at the front of the 
        # queue is the largest element of 
        # previous window, so print it 
		while Q and Q[0] <= i-k:# Remove the elements which are  
        # out of this window 
			Q.popleft()
		while Q and arr[i] >= arr[Q[-1]]:# Remove all elements smaller than 
        # the currently being added element  
        # (Remove useless elements)
			Q.pop()
		Q.append(i)# Add current element at the rear of Qi 
	result.append(Q[0])# Print the maximum element of last window 
	return result
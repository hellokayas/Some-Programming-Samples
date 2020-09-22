import math
import sys

#Given two sorted integer arrays A and B, merge B into A as one sorted array in linear time.

def merge(A,B):# just maintain two pointers to see which one is smaller and append accordingly to the result
	result = []#but this is not done in place though the time is linear in the size of arrs
	n = len(A)
	m = len(B)
	i,j = 0,0
	while(i<n and j<m):# same approach can be used to find the common elems from two sorted arrs
		if A[i] <= B[j]:
			result.append(A[i])
			i += 1
		else:
			result.append(B[j])
			j += 1
	if (i == n):
		while(j<m):
			result.append(B[j])
			j += 1
	if (j == m):
		while(i<n):
			result.append(A[i])
			i += 1
	return result# for inplace, the insert(index,elem) function will do the job.

#Given an array A of n integers, find three integers in A such that the sum is closest to a given target and 
# return the sum of the three integers.

def closest(A,target):
	n = len(A)
	A.sort()
	result = float("-inf")
	for i in range(0,n-2):# we fix the smallest elem and vary the larger in the sorted arr, so leave some space for the last two elem
		left = i+1# starting from just larger than smallest
		right = n-1
		while(left < right):
			s = A[i] + A[left] + A[right]# if we need to find 3 tuple which sum to 0, then fix smallest i and search j,k such that 
			if s == target:# A[j] + A[k] = -A[i]
				return target
			if abs(s - target) < abs(result - target):# this triplet sum is closer
				result = s
			if s > target:
				right -= 1
			if s < target:
				left += 1
	return result

#You are given an array of N non-negative integers. Considering each array elem as the edge length of some line segment, count the
# number of triangles which you can form using these array values.

def triangles(A):#First we sort the array of side lengths. So since Ai < Aj < Ak where i < j < k, therefore it is sufficient 
#to check Ai + Aj > Ak to prove they form a triangle.
	n = len(A)
	A.sort(reverse = True)# put the biggest value in the front
	answer = 0
	for i in range(0,n-2):
		third_side = A[i]
		if A[i] == 0:# degenerate case
			break
		j = i+1
		k = n-1
		while(j < k):# a triangle is possible, infact possible for all j as k ranges till j, since A[k] is going larger and hence sum > A[i] 
			if A[j] + A[k] > A[i]:
				answer += k-j# so add all those cases together in one go
				j += 1# increase j and check again, these triangles are different since now A[j] is smaller as j moves right
			else:
				k -= 1# the LHS is smaller , so make it larger moving k to right, as the arr is in descending order
	return answer

#Given a sorted array, remove the duplicates in place such that each element can appear atmost twice and return the new length.
#Do not allocate extra space for another array, you must do this in place with constant memory.

def removeduplicate(A):
	n = len(A)
	count = 0
	for i in range(0,n-2):
		if (A[i] == A[i+1] and A[i] == A[i+2]):# all eqaul elems are clustered together. We are taking the last two equal ones from the
			continue# the cluster and putting them in the front and count pointer marks the number too.
		else:# the front of the arr is the updated part, rest are moved to the back. At any moment, A[:count+1] is the updated part
			A[count] = A[i]
			count += 1
	return count

# given an arr which contains only elems 0,1,2 sort them in linear time.

def specialsort(A):
	B = [0,0,0]
	for i in A:# supersmart technique! This is working since the indexes in B and the elems in A match.
		B[i] += 1# what to do if elems in A are 0,2,5,9 ? Then map it to 0,1,2,3 and then use the same technique.
	return [0]*B[0] + [1]*B[1] + [2]*B[2]

#You are given with an array of 1s and 0s. And you are given with an integer M, which signifies number of flips allowed.
#return the indices of maximum continuous series of 1s in order.

def maxone(A,M):# M is the num of flips allowed and A is the arr
	n = len(A)# i and j are the pointers that run and besti and bestj are the upto date pointers showing what is the max run of ones
	i,j,besti,bestj = 0,0,0,0# found till now. 
	for j in range(n):# i runs behind j, i will be required to count the gap = largest run of 1's
		if A[j] == 0:# if 0 then we have to flip it so one flip gone from M flips
			M -= 1# as soon as we run out of M flips we move the lower pointer i to be in the bound of possible M flips
			while(i <= j and M < 0):# we want to move i to right
				if A[i] == 0:# but that is only if we encounter a 0, 
					B += 1
				else:# doesnt matter if 1 since no flips necessary
					B += 0
				i += 1# just move on after B is suitably incremented
		j += 1# when we have not found 0 then just move right, nothing to care about
		if j-i > bestj - besti:
			besti,bestj = i,j# the part where we maintain the updates
	return list(range(besti,bestj))

#You are given 3 arrays A, B and C. All 3 of the arrays are sorted. there exist i, j, k such that :
# max(abs(A[i] - B[j]), abs(B[j] - C[k]), abs(C[k] - A[i])) is minimized. Return that minimum

def minimize(A,B,C):
	a,b,c = len(A),len(B),len(C)
	x,y,z = 0,0,0# these are the pointers which will run
	maximum = float("inf")
	minimum = float("-inf")
	answer = float("inf")
	while (x < a and y < b and z < c):
		maximum = max(A[x],B[y],C[z])
		minimum = min(A[x],B[y],C[z])#If new result is less than current result, change it to the new result.
		answer = min(answer,maximum-minimum)# Compute max(X, Y, Z) - min(X, Y, Z)
		if answer == 0:# it wont get better than this!
			break
		if minimum == A[x]:# Increment the pointer of the array which contains the minimum
			x += 1#we increment the pointer of the array which has the minimum, because our goal is to decrease the difference. 
		elif minimum == B[y]:#Increasing the maximum pointer is definitely going to increase the difference
			y += 1#Increase the second maximum pointer can potentially increase the difference
		else:#however, it certainly will not decrease the difference
			z += 1
	return answer

#Given a list heights of n non-negative integers a1, a2, ..., an. where each represents a point at coordinate (i, ai).
#'n' vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0),
#Find two lines, which together with x-axis forms a container, such that the container contains the most water.
#same problem of finding max area of a rectangle under histogram.

def maxarea(heights):# the area is always min(ai,aj)*(j-i) for j>i as i,j ranges over the whole arr
	n = len(heights)
	left = 0#When you consider a1 and aN, then the area is (N-1) * min(a1, aN). Start with that.
	right = n-1# The base (N-1) is the maximum possible.
	area = 0# This implies that if there was a better solution possible, it will definitely have height greater than min(a1, aN)
	while(left < right):#This means that we can discard min(a1, aN) from our set and look to solve this problem again from the start
		area = max(area,min(heights[left],heights[right])*(right-left))
		if heights[left] <= heights[right]:
			left += 1#If a1 < aN, then the problem reduces to solving the same thing for a2, aN.
		else:
			right -= 1#Else, it reduces to solving the same thing for a1, aN-1
	return area

#Given an array A of N non-negative numbers and you are also given non-negative number B.
#You need to find the number of subarrays in A having sum less than B. 

def maxsubarr(A,B):
	n = len(A)
	start, end, ans, bound = 0,0,0,0# bound should not be crossed as the sum of elem in a subarr, start and end are the pointers
	while(start < n and  end < n):
		if bound < B:
			end += 1# we can add elem to our arr
			bound += A[end]# bound increases by the added elem
			ans += end - start# there are thses many arr in that range
		if bound >= B:# we have to remove elem from the arr
			bound -= A[start]# increase the start to start from a new elem as a new arr
			start += 1# do this until bound < B
	return ans

#Given an array A of positive integers,call a (contiguous,not necessarily distinct) subarray of A good if the number of different
# integers in that subarray is exactly B. Return the number of good subarrays of A.

def goodsubarr(A,B):#Function to return the count of subarrays
# with exactly B distinct elements
	n = len(A)
#So the idea is to find the count of subarrays with at most B different integers, let it be C(B), and the count of subarrays with at
# most (B - 1) different integers, let it be C(B - 1) and finally take their difference, C(B) – C(B – 1) which is the required answer.
	def atmost(A,B):# Function to return the count of subarrays with at most K distinct elements
		count,left,right = 0,0,0# count gives the ans, left & right are the pointers
		map = {}## Map to keep track of number of distinct elements in the current window
		while(right < n):
			if A[right] not in map:
				map[A[right]] = 0
			map[A[right]] += 1## Calculating the frequency of each
        # element in the current window
			while(len(map) > B):## Shrinking the window from left if the
        # count of distinct elements exceeds B
				if A[left] not in map:
					map[A[left]] = 0
				map[A[left]] -= 1
				if map[A[left]] == 0:
					del map[A[left]]
				left += 1
			count += right-left + 1## Adding the count of subarrays with at most
        # B distinct elements in the current window
			right += 1
		return count

	return atmost(A,B)-atmost(A,B-1)
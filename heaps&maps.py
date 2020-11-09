import string
import math
import sys
from collections import Counter
from collections import defaultdict
from heapq import *
import operator as op
from functools import reduce

#Given two arrays A & B of size N each.
#Find the maximum N elements from the sum combinations (Ai + Bj) formed from elements in array A and B

'''
Sort both arrays array A and array B. 
Create a max heap i.e priority_queue in C++ to store the sum combinations along with the indices of elements from both arrays A and B which 
make up the sum. Heap is ordered by the sum. 
Initialize the heap with the maximum possible sum combination i.e (A[N – 1] + B[N – 1] where N is the size of array) and with the indices
 of elements from both arrays (N – 1, N – 1). The tuple inside max heap will be (A[N-1] + B[N – 1], N – 1, N – 1). Heap is ordered by first
 value i.e sum of both elements. 
Pop the heap to get the current largest sum and along with the indices of the element that make up the sum. Let the tuple be (sum, i, j). 
Next insert (A[i – 1] + B[j], i – 1, j) and (A[i] + B[j – 1], i, j – 1) into the max heap but make sure that the pair of indices
 i.e (i – 1, j) and (i, j – 1) are not already present in the max heap. To check this we can use set in C++. 
Go back to 4 until N times.'''

def maxpair(A,B):
	N = len(A)
	visited = set()
	A = sorted(A, reverse = True)
	B = sorted(B, reverse = True)# we do it in the reverse way so that we can access A[0] and do not have to access A[N-1]
	result = []
	heap = []
	visited.add((0,0))
	heapq.heappush(heap,(-(A[0] + B[0]), (0,0)))
	for _ in range(N):
		mysum, (iA,iB) = heapq.heappop(heap)
		result.append(-mysum)

		tuple1 = (iA + 1, iB)
		if iA < N-1 and tuple1 not in visited:
			visited.add(tuple1)
			heapq.heappush(heap,(-(A[iA+1] + B[iB]), (iA+1,iB)))
		tuple2 = (iA, iB + 1)
		if iB < N-1 and tuple2 not in visited:
			visited.add(tuple2)
			heapq.heappush(heap,(-(A[iA] + B[iB+1]), (iA+1,iB)))
	return result

def ncr(n, r):
    r = min(r, n - r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n - r, -1))
    denom = reduce(op.mul, range(1, r + 1))
    return numer // denom

#Given N bags, each bag contains Bi chocolates. There is a kid and a magician. In one unit of time, kid chooses a random bag i, 
#eats Bi chocolates, then the magician fills the ith bag with floor(Bi/2) chocolates.
#Find the maximum number of chocolates that kid can eat in A units of time

# Just greedily choose the largest bag always and update the heap so that when we pop, we always choose the largest bag

def maxchoc(A,B):
	h = [-b for b in B]
	ans = 0
	heapify(h)
	for _ in range(len(B)):
		choc = heappop(h)
		ans += choc
		heappush(h,choc//2)
	return ans

#Merge k sorted linked lists and return it as one sorted list

#Now one way will be to connect all the nodes into one single LL in no particular order and then use mergesort of LL to
# sort the whole list in O(nlogn) time. This does not use the info that each of the k lists is sorted.

#Second approach will be to use MinHeap. This will be done in O(nlogk) time and O(k) space to store the minheap with k nodes
#construct a minheap of size k and insert the first node of each list into it. Then we pop the root node from the heap and insert
# the next node from the same list as popped node. We repeat this until the heap is exhausted.

# the third way is the best: Divide and Conquer. this has the same time complexity but gets rid of the O(k) space. We can merge
# two sorted LL in O(n) time and O(1) space. The idea is to pair up K LL and merge each pair in linear time and const space. So after
# first cycle k/2 lists are left of length 2*N, after 2nd cycle, k/4 lists of length 4*N. Repeat until there is only one list left.

# Takes two lists sorted in increasing order, and merge their nodes together to make one big sorted list which is returned
def sortedMerge(a, b):

	# Base cases
	if a is None:
		return b
	elif b is None:
		return a

	# Pick either a or b, and recur
	if a.data <= b.data:
		result = a
		result.next = sortedMerge(a.next, b)
	else:
		result = b
		result.next = sortedMerge(a, b.next)

	return result

# The main function to merge given k sorted linked lists. A is the list of k lists, so A is of form [[]].
def mergeKLists(A, k):

	last = k - 1
	# repeat until only one list is left
	while last:
		(i, j) = (0, last)

		# (i, j) forms a pair
		while i < j:
			# merge List i with List j and store
			# merged list in List i
			A[i] = sortedMerge(A[i], A[j])

			# consider next pair
			i = i + 1
			j = j - 1

			# If all pairs are merged, update last
			if i >= j:
				last = j

	return A[0]

# Implementation of LRU cache
from collections import deque
class LRUCache:

    # @param capacity, an integer
    def __init__(self, capacity):
        self.capacity = capacity
        self.dic = {}
        self.q = deque()
        

    # @return an integer
    def get(self, key):
        if key in self.dic:
            self.q.remove(key)
            self.q.appendleft(key)
            return self.dic[key]
        return -1
        

    # @param key, an integer
    # @param value, an integer
    # @return nothing
    def set(self, key, value):
        if key in self.dic:
            self.q.remove(key)
        elif self.capacity == len(self.dic):
            keyToRemove = self.q.pop()
            del self.dic[keyToRemove]
        self.q.appendleft(key)
        self.dic[key] = value

#You are given an array of N integers, A1, A2 ,..., AN and an integer B. Return the of count of distinct numbers in all windows
# of size B. Formally, return an array of size N-B+1 where i'th element in this array contains number of distinct elements in 
# sequence Ai, Ai+1 ,..., Ai+B-1

def countDistinct(arr, k, n): 
  
    # Creates an empty hashmap hm  
    mp = defaultdict(lambda:0) 
  
    # initialize distinct element  
    # count for current window  
    dist_count = 0
  
    # Traverse the first window and store count  
    # of every element in hash map  
    for i in range(k): 
        if mp[arr[i]] == 0: 
            dist_count += 1
        mp[arr[i]] += 1
  
    # Print count of first window  
    print(dist_count) 
      
    # Traverse through the remaining array  
    for i in range(k, n): 
  
        # Remove first element of previous window  
        # If there was only one occurrence,  
        # then reduce distinct count.  
        if mp[arr[i - k]] == 1: 
            dist_count -= 1
        mp[arr[i - k]] -= 1
      
    # Add new element of current window  
    # If this element appears first time,  
    # increment distinct element count  
        if mp[arr[i]] == 0: 
            dist_count += 1
        mp[arr[i]] += 1
  
        # Print count of current window  
        print(dist_count) 

'''here is the idea of the solution
Create an empty hash map. Let the hash map be hM.

Initialize the count of distinct element as dist_count to 0.

Traverse through the first window and insert elements of the first window to hM. The elements are used as key and their counts
 as the value in hM. Also, keep updating dist_count

Print distinct count for the first window.

Traverse through the remaining array (or other windows).

Remove the first element of the previous window.
If the removed element appeared only once, remove it from hM and decrease the distinct count, i.e. do “dist_count–“
else (appeared multiple times in hM), then decrement its count in hM

Add the current element (last element of the new window)
If the added element is not present in hM, add it to hM and increase the distinct count, i.e. do “dist_count++”
Else (the added element appeared multiple times), increment its count in hM
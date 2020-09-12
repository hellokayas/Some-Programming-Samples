import math
from bisect import bisect_left

def bin_search(a,x):
	i = bisect_left(a,x)#finds the left most pos of elem x in sorted arr a
	if i:
		return i-1
	else:
		return -1

#You are in an infinite 2D grid where you can move in any of the 8 directions.
#You are given a sequence of points and the order in which you need to cover 
#the points.. Give the minimum number of steps in which you can achieve it. You start from the first point.

def coverpts(A,B):
	ans = 0
	for i in range(len(A)):
		ans += max(abs(A[i] - A[i-1]), abs(B[i] - B[i-1]))
	return ans

#Given an array of intervals in sorted order and a new interval, return a sorted array after merging the interval

def mergeinter(intervals,newinter):
	n = len(intervals)
	start = newinter[0]# we mark the start and end of the new interval to be merged
	end = newinter[1]
	right,left = 0,0
	while right < n:# we track where this new interval belongs, i.e. how many interval are to the left of it and how many are to the right
		if start <= intervals[right][1]:# we find the first interval before which it fits
			if end < intervals[right][0]:# this checks if the interval is disjoint and lies between two given intervals
				break# in this case we have nothing to do and go to line 29 directly
			start = min(start,intervals[right][0])# if it intersects with the given intervals then we just update and merge the ends
			end = max(end,intervals[right][1])
		else:# counting how many to the left continuing from line 20
			left += 1 
		right += 1# moving right to find the fit continuation of line 20 and even if we merge in line 25, we go to the next interval before
	return intervals[:left] + [(start,end)] + intervals[right:] # we return starting from the next interval

#Given a collection of intervals, merge all overlapping intervals and return sorted list of disjoint intervals.

def merge(I):
	I.sort(key:lambda i:i[0])# sorting according to the start of all intervals
	res = []# we start from the last of the given arr of lists and check the ends of the intervals and merge from the end
	for i in I:
		if not res or res[-1][0] < i[1]:# if res is empty then we put an elem in it from I
			res.append(i)# if there is no overlap, just add it
		else:
			res[-1][1] = max(i[1], res[-1][1])# here we merge from the end so that res remains sorted
	return res

#Find the contiguous subarray within an array A, which has the largest sum.

def maxsubarr(A):# here we have to observe that if (A[i],...,A[j]) is an optimal soln then this does not have a prefix,
	currsum = 0# say (A[i],...,A[i+10]) whose sum is negative. Similarly, we should always include prefixes with +ve sum
	maxsum = A[0]#So we should always start from beginning to include as much +ve prefix possible
	n = len(A)
	for i in A[1:]:
		currsum = currsum + i
		maxsum = max(maxsum,currsum)
		if currsum <0:# but if the sum of prefixes at any stage becomes negative, we drop that prefix immediately as this cannot
			currsum = 0# be optimal and start from 0 again for currsum, keeping a global maximum till recorded
	return maxsum

#Given an integer array A, find if an integer p exists in the array such that the number of integers greater than or equal to p in the
# array equals to p. Return 1 if any such integer p is found else return -1.

def solve(A):
	n = len(A)
	A.sort()# sort the list and then check how many numbers greater than or eqaul to a number at each index
	for i in range(n-1):
		if A[i] == A[i+1]:# if equal then go to the next number to find out how many really greater
			continue
		if A[i] == n - i -1: # num of integers greater than p = A[i]
			return 1
	if A[n-1] == 0:#check, might not be true since greater is reqd not greater or equals!!
		return 1
	return -1

#Given a list of non negative integers, arrange them and then join them together such that they form the largest number.

def largestnum(A):
	A = map(str,A)# first we map the list of integers to strings only to do line 75 later
	res = "".join(sorted(A, cmp = lambda a,b:cmp(a+b,b+a), reverse = True))#sort acc to if a+b >, =, < b+a after joining strings sidebyside
	#we compare two numbers XY (Y appended at the end of X) and YX (X appended at the end of Y).
	#If XY is larger, then, in the output, X should come before Y, else Y should come before X.
	res = res.lstrip("0")# if any leading 0's are formed due to concatenation
	if not res:# if everything is empty
		return 0
	else return res

#Given a positive integer n and a string s consisting only of letters D or I, you have to find any permutation of first n positive integer
# that satisfy the given input string. D means the next number is smaller, while I means the next number is greater.

def findperm(A,B): #A is the string and B is the integer
	mn,mx = 1,B
	res = []
	for x in A:
		if x == "D":#next num to be smaller
			res.append(mx)
			mx = mx -1#going from back
		if x == "I":#next num to be larger
			res.append(mn)
			mn = mn +1#going from front
	res.append(mn)#append(mx) will do too, only one number left
	return res

#Given a non-negative number represented as an array of digits, add 1 to the number

def addone(A):
	carry = 1
	A.insert(0,0)#inserting two 0's incase there is overflow in the most significant digit
	for i in range(len(A)-1,-1,-1):#going backwards and writing in place
		D = carry + A[i]#consider this number
		carry = D//10# this is what carry should be
		A[i] = D % 10# this is what should be put back to place after addition and considering the carry
	while len(A) > 0 and A[0] == 0:# removing all zeroes if they appear in the front
            del A[0]
	return A

#Find out if any integer occurs more than n/3 times in the given array A in linear time .

def repeatednum(A):# the idea i if we find three distinct elem in A, removing all 3 of them wont change the ans. Sp actually we are making
	pos1,pos2,ct1,ct2 = 0,0,0,0# 3 tuples of these disctinct numbers and removing them. Whatever left will be the reqd value.
	for x in A:#Start with two empty candidate slots(pos) and two counters(ct) set to 0.
		if pos1 == x:#if it is equal to either candidate, increment the corresponding count
			ct1 += 1
		if pos2 == x:#if it is equal to either candidate, increment the corresponding count
			ct2 += 1
		elif ct1 == 0:#else if there is an empty slot (i.e. a slot with count 0), put it in that slot and set the count to 1
			pos1, ct1 = x,1
		elif ct2 == 0:#else if there is an empty slot (i.e. a slot with count 0), put it in that slot and set the count to 1
			pos2, ct2 = x,1
		else:#else reduce both counters by 1. this is the step of removing three distinct elems from A.
			ct1 -= 1
			ct2 -= 1

	ct1,ct2 = 0,0
	for y in A:#At the end, make a second pass over the array to check whether the candidates really do have the required count.
		if y == pos1:
			ct1 += 1
		if y == pos2:
			ct2 += 1
	if ct1 > len(A)/3:#If there is a value that occurs more than n/3 times then it will be in a slot
		return pos1
	if ct2 > len(A)/3:#but you don't know which one it is. So need to check both the slots.
		return pos2
	return -1#if the numbers in the slot do not satisfy, then there is none for sure.

#A hotel manager has to process N advance bookings of rooms. His hotel has K rooms. 
#Bookings contain an arrival date and a departure date. Enough rooms for demand?

def hotel(A,D,K):#arrival array is A and departure times are in array D. K is the number of rooms available
	events = [(t,1) for t in A] + [(t,0) for t in D]# combined list of arrival and departures with arr marked 1 and dep marked 0
	events.sorted()# sort everything acc to time
	guest = 0# counter of guests at any time
	for event in events:
		if event[1] == 1:#if arrival then increasing the count of guests
			guest += 1
		else:# if dep then  decrease the count of guests
			guest -= 1
	if guest > K:# counter keeps track of max guests arrived at any time, so if that max > K then not enough rooms
		return -1
	return 1# no problem!

#You are given an n x n 2D matrix. Rotate the image by 90 degrees inplace.(clockwise)

def rotate(A):
	n = len(A)
	for i in range(n):# We plan to take the transpose of the matrix first
		for j in range(i,n):# if we take the full range then the matrix will remain as it is as the elem flip back to their original pos
			A[i][j], A[j][i] = A[j][i], A[i][j]
	for i in range(n//2):# running over columns
		for j in range(n):# running over rows
			A[j][i], A[j][n-1-i] = A[j][n-1-i], A[j][i]#exchanging the columns from front and back along each row
	return A#transpose + flipping colums = 90 deg rot clockwise!

#You are given an integer array B containing n integers. Count the number of ways to split all the elements of the array into
# 3 contiguous parts so that the sum of elements in each part is the same.

def split(B):
	n = len(B)
	total = sum(B)
	if total % 3 == 0:# the arr B must be divided into 3 equal parts o/w finding such 3 parts not possible
		target = total % 3# the value each of the 3 parts must have
	else:
		return 0# the arr B cannot be divided into such 3 parts
	ans = 0
	first_end = 0# the last value of the first part of the 3 parts of the arr
	cumul_sum = 0#gathering sum of the elems beginning from start
	for i in range(n-1):#should leave out the last elem since ther must remain a 3rd part, 3nd part cannot end at B[n-1]
		cumul_sum = cumul_sum + B[i]
		if cumul_sum == target:# as many times we hit the target starting from B[0], we need to calculate the number of times
			first_end = first_end + 1
		if cumul_sum == 2 * target:# this is the moment when we have found a possible last elem for the middle part
			ans = first_end + ans# all possible combinations of choosing a last elem for first part and one for end of middle part
	return ans

#Find the Minimum length Unsorted Subarray, sorting which makes the complete array sorted

def unsorted(arr,n):# n = len(arr)
	end = n-1
	for start in range(n):#Scan from left to right and find the first element which is greater than the next element, call it start
		if arr[start] > arr[start + 1]:
			break#as soon as we have found it
	if start == n - 1:#last elem
		return -1# the list is already sorted
	while (end >= 0):#Scan from right to left and find the first element (first in right to left order)
		if arr[end] < arr[end - 1]:# which is smaller than the next element
			break#as soon as we have found it
		end -= 1#just decrement when we are scanning from right to left
	if end == 0:# first elem
		return -1# already sorted
	newarr = arr[start+1:end+1]#the essential unsorted array to be dealt with
	ourmax = max(newarr)
	ourmin = min(newarr)
	for i in range(0,start+1):#Find the first element (if there is any) in arr[0..s-1] which is greater than ourmin, 
		if ourmin < arr[i]:
			start = i#change start to index of this element
			break#as soon as we have found this point there is no need to worry aboout the left anymore, everything is sorted
	j = n-1#start from the right
	while(j > end):#Find the last element (if there is any) in arr[e+1..n-1] which is smaller than ourmax,
		if ourmax > arr[j]:
			end = j# change e to index of this element
			break#as soon as we have found this point there is no need to worry aboout the right anymore, everything is sorted
		else:
			j = j-1
	return arr[start,end+1]# the reqd aary to be sorted to get whole arr sorted

#Given an array, find the maximum j – i such that arr[j] >= arr[i].

def maxindexdiff(arr,n):# n is the length of the arr
	maxdiff = 0
	Lmin = [] * n
	#Construct LMin[] such that  
    # LMin[i] stores the minimum  
    # value from (arr[0], arr[1],  
    # ... arr[i])  
	Rmax = [] * n
	# Construct RMax[] such that  
    # RMax[j] stores the maximum  
    # value from (arr[j], arr[j + 1], 
    # ..arr[n-1])  
	Lmin[0] = arr[0]
	for i in range(n):# once we find the minimum we have all the min in all places. all elements on left of LMin[i]
		Lmin[i] = min(arr[i], Lmin[i-1])# are greater than or equal to LMin[i]
	Rmax[n-1] = arr[n-1]
	for j in range(n-2,-1,-1):#Once we find the maximium we put that in all the places all the way upto the start of Rmax
		Rmax[j] = max(arr[j], Rmax[j+1])#all elem on right of Rmax[j] are larger than Rmax[j]

	i,j = 0,0
	while(i<n and j<n):
		if Lmin[i] < Rmax[j]:#checks the arr[i] <= arr[j] here
			maxdiff = max(maxdiff, j-i)# if holds then only care to look at j-i
			j = j+1#go a little greater since as we go right in Rmax, we are having bigger elems from arr and also increasing the index
		else:
			i = i+1#if the condition doesnt hold go a little smaller since smaller elems are on right of Lmin. DOnt wanna do it usually
			# as we are increasing the index in arr and so the diff is going smaller.
	return maxdiff

#Given an integer array A of size N. You can pick B elements from either left or right end of the array A to get maximum sum.
#Find and return this maximum possible sum.

def solve(self, A, B):#greedily choosing the largest from each side at any instant does not work. We have to consider the full sum.
        sumA , sumB= sum(A[:B]),0#sumA is the sum_left i.e summed upto B elems from the left
        maxi  =sumA#we need to update this, currently guessing sumA is the maximum, we have to see about the right of A now
        i,j = B-1, len(A)-1
        for _ in range(B):
            sumA -= A[i]#subtract one by one from the inside and
            sumB += A[j]# start adding from the right to see if the maxsum increases or not 
            maxi = max(maxi , sumA + sumB)# and update it if elem from the right if taken increases the sum
            i-=1
            j-=1
        return maxi

# function to find maximum sum k x k sub-matrix in a given matrix mat(NxN) in time O(N**2)
def findMaxSumSubMatrix(mat, k):#can refer to https://www.techiedelight.com/find-maximum-sum-submatrix-in-given-matrix/

	# M x N matrix
	(M, N) = (len(mat), len(mat[0]))

	# pre-process the input matrix such that sum[i][j] stores
	# sum of elements in matrix from (0, 0) to (i, j)
	sum = [[0 for x in range(N)] for y in range(M)]
	sum[0][0] = mat[0][0]

	# pre-process first row
	for j in range(1, N):
		sum[0][j] = mat[0][j] + sum[0][j - 1]

	# pre-process first column
	for i in range(1, M):
		sum[i][0] = mat[i][0] + sum[i - 1][0]

	# pre-process rest of the matrix
	for i in range(1, M):
		for j in range(1, N):
			sum[i][j] = mat[i][j] + sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1]

	max = float('-inf')

	# find maximum sum sub-matrix

	# start from cell (k - 1, k - 1) and consider each
	# sub-matrix of size k x k
	for i in range(k - 1, M):
		for j in range(k - 1, N):

			# Note (i, j) is bottom right corner coordinates of
			# square sub-matrix of size k

			total = sum[i][j]
			if i - k >= 0:
				total = total - sum[i - k][j]

			if j - k >= 0:
				total = total - sum[i][j - k]

			if i - k >= 0 and j - k >= 0:
				total = total + sum[i - k][j - k]

			if total > max:
				max = total
				p = (i, j)

	# returns coordinates of bottom right corner of sub-matrix
	return p

#Given an integer array A of size N. You need to count the number of special elements in the given array.
#A element is special if removal of that element make the array balanced.
#Array will be balanced if sum of even index element equal to sum of odd index element.
def solve(A):
        n = len(A)
        odd = 0
        even = 0
        leftOdd = [0] * n#prefixsum for odd pos
        rightOdd = [0] * n# suffix sum for odd pos
        leftEven = [0] * n# prefix sum for even pos
        rightEven = [0] * n# suffix sum for even pos
        for i in range(n):# now making those prefixa nd suffix sum arr
            leftOdd[i] = odd#prefix_odd
            leftEven[i] = even#prefix_even
            if(i%2 == 0):
                even += A[i]
            else:
                odd += A[i]
        
        odd = 0
        even = 0
        for i in range(n-1, -1, -1):#now making the suffix sums
            rightOdd[i] = odd
            rightEven[i] = even
            if(i%2 == 0):
                even += A[i]#suffix_even
            else:
                odd += A[i]
        
        ans = 0
        for i in range(n):#when we remove an elem the prefix and suffix sums for the even and odd pos flips
            if(leftOdd[i] + rightEven[i] == leftEven[i] + rightOdd[i]):#this is what actually happens when we remove an elem from the arr
                ans += 1
            
        return ans


#Given an array A containing N integers. You need to find the maximum sum of triplet ( Ai + Aj + Ak )
# such that 0 <= i < j < k < N and Ai < Aj < Ak. If no such triplet exist return 0.

def solve(A):
	n = len(A)
	suffix = [0]* n# here we are just building the suffix arr to find the largest elem in arr which is after A[i]
	suffix[n-1] = A[n-1]
	for i in range(n-2,-1,-1):
		suffix[i] = max(suffix[i+1],A[i])# building the arr and we will use it in line 380
	ans = 0
	sorted_arr = []# this is the arr we maintain always in a sorted fashion which helps us to find the smallest elem before
	# an elem A[i]. For that we use binsearch in that sorted portion before A[i]. for updating the arr, insert is the suitable func. 
	sorted_arr.append(A[0])
	for i in range(1,n-1):# why upto n-1? if upto n, i goes to n-1 and suffix[n] will be needed. suffix[n-1] is the last
		res = bin_search(sorted_arr,A[i])# here findinf the elem before A[i] which is max and smaller than A[i]
		if res != -1 and A[i] < suffix[i+1]:# != -1 means we have found the smallest beore A[i] and second cond is Aj < Ak in line 366
			temp = sorted_arr[res] + A[i] + suffix[i+1]# this is the sum we need to see
			ans = max(ans, temp)# updating the max
		sorted_arr.insert(res+1,A[i])# suitably maintaining the arr in a sorted fashion. this just adds a new elem to the list like
		#append if not found. If found then insert in the right pos as reported by bisect_left
	return ans
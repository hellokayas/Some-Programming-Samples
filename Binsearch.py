import math
import sys

# find first and last occurances(two different functions) of a number in a given sorted array using binsearch.

def first(arr,high,low,x):#high is the last elem and low is the first elem in arr
	while(high >= low):# when this breaks binsearch stops, elem not found
		mid = low + (high-low)//2
		if ((mid == 0 or x > arr[mid-1]) and x == arr[mid]):#x == arr[mid] and > arr[mid-1] means first occurence
			return mid
		elif x < arr[mid]:
			return first(arr,mid-1,low,x)# recurse
		else:
			return first(arr,high,mid+1,x)# recurse
	return -1

def last(arr,high,low,x):# everything same as above
	n = len(arr)
	while(high>= low):
		mid = low + (high-low)//2
		if((mid == n-1 or x < arr[mid+1]) and x == arr[mid]):#x == arr[mid] and < arr[mid+1] means last occurence
			return mid
		elif x < arr[mid]:
			return first(arr,mid-1,low,x)
		else:
			return first(arr,high,mid+1,x)
	return -1

# given int A, return its sqrt. If not perfect sq return floor of sqrt

def sqrt(A):
	return first(A,0,A)
	def first(high,low,A):
		while(high >= low):# exact same idea of the first functiona s above
			mid = low + (high-low)//2
			if ( A == mid**2):
				return mid
			elif A < mid**2 and A <= (mid-1)**2:
				return first(mid-1,low,A)
			elif A< mid**2 and A > (mid-1)**2:# this is for the floor of sqrt
				return mid -1
			elif A > mid**2 and A >= (mid+1)**2:
				return first(high,mid+1,A)
			else:
				return mid# this is for the floor of the sqrt

# given x, n and d, find (x^n % d)

def power(x,n,d):
  x = x % d
  if n == 1:
    return x % d
  while (n > 1):
    if n % 2 == 0:
      return ((power(x, n//2,d) %d)**2) % d
    if n % 2 == 1:
      return ((power(x, n//2,d) %d)**2 * x) % d

#Find median in row wise sorted matrix M with r rows and c columns such that r*c is odd

def matmedian(M,r,c):
	matmax,matmin = 0,0
	for i in range(r):
		if M[i][0] <= matmin:# for calculating the min the matrix looking at the first column is enough
			matmin = M[i][0]
	for i in range(r):
		if M[i][c-1] >= matmax:# for calculating the min the matrix looking at the last column is enough
			matmax = M[i][c-1]

	desired = (r*c + 1)//2#If we consider the N*M matrix as 1-D array then the median is the element of 1+r*c/2 th element.
	#Then consider x will be the median if x is an element of the matrix and number of matrix elements ≤ x equals 1 + r*c/2
	while (matmin < matmax):# now starting bin search. As the matrix elements in each row are sorted then you can easily 
	#find the number of elements in each row less than or equals x
		how_many_smaller = 0# counts how many smaller than a given x in the matrix and then will check if less than desired or not
		mid = matmin + (matmax - matmin)//2
		for i in range(r):
			j = last(M[i],M[i][c-1],M[i][0],mid)#Then first find the minimum and maximum element from the N*M matrix.
			# Apply Binary Search on that range and run the above function for each x
			how_many_smaller += j
		if how_many_smaller < desired:#num of elem in matrix ≤ x is 1 + r*c/2 and x contains in that matrix then x is the median
			matmin = mid +1
		else:
			matmax = mid
	return matmin

#Searching in a sorted and rotated array. The interesting property of a sorted + rotated array is that when you divide it into two halves,
#atleast one of the two halves will always be sorted. If mid happens to be the point of rotation them both left and right
# sub-arrays will be sorted.

def rotarr_search(arr,key):
	start,end = 0,len(arr)-1
	while(start <= end):
		mid = start + (end-start)//2
		if arr[mid] == key:
			return mid
		elif (arr[start] <= key < arr[mid]) or (arr[start] > arr[mid] and not(arr[mid] < key <= arr[end])):#We can easily know which half
		# is sorted by comparing start and end element of each half.Once we find which half is sorted we can see if the key is present in
		# that half - simple comparison with the extremes. If the key is present in that half we recursively call the function
		# on that half else we recursively call our search on the other half.
			end = mid - 1
			# We choose left half if either : 
                #    * left half is sorted and B in this range
                #    * left half is not sorted, 
                #      but B isn't in the sorted right half.
		else:
			start = mid+1
	return -1

#There are two sorted arrays A and B of size x and y respectively( x < y). Find the median of the two sorted arrays in Log time

def findmedian(A,B):
	if len(A) > len(B):
		return findmedian(B,A)# always ensuring that we do the binsearch on the shorter arr
	x = len(A)
	y = len(B)
	start = 0# start and end of shorter arr
	end = x
	while (start <= end):
		partition_x = (start + end)//2# the mid of smaller arr, partition_x is an index
		partition_y = (x+y+1)//2 - partition_x# the suitable partition of larger arr to divide the arrs into equal halves
		if partition_x == 0:# if there is nothing on the left
			left_x = None
		if partition_x == x:# if there is nothing on the right
			right_x = sys.maxint# +inf
		if partition_y == 0:
			left_y = None# this is -inf similar to the case for smaller arr
		if partition_y == y:
			right_y = sys.maxint

		if (left_x <= right_y) and (left_y <= right_x):# all the elems on left are smaller than all the elems on right is ensured by
		 #checking on the extremes only since arrs sorted. Also, the partition always makes equal halves, so found the right spot.
			if (x+y) % 2 == 0:
				return (max(left_x,left_y) + min(right_x,right_y))/2.0
			else:
				return max(left_x,left_y)# if the num of elems is odd
		elif left_x > right_y:# if we have come more towards right of smaller arr, then move left on smaller arr
			end = partition_x -1
		else:# if we have come more to the left
			start = partition_x + 1

#The painter’s partition problem. arr contains different lengths to be painted. P is the num of painters available.

def partition(arr,P):
	high = sum(arr)
	low = max(arr)
	while (low < high):
		mid = low + (high-low)//2
		reqd_num_painters = isfeasible(arr,mid)# min number of painters returned
		if reqd_num_painters <= P:# if that is less than P, we can do better, assign workload to more painters, so done faster
			high = mid
		else:# we have to appoint more painters, searching for a better partition
			low = mid + 1
	return low

	def isfeasible(arr,mid):#It finds the minimum number of segments we can get with mid as max workload for a painter
		painters = 1
		workload = 0
		for i in arr:
			workload += i
			if workload > mid:# so as it crosses, appoint a new painter
				workload = i
				painters += 1
		return painters# min number of segments = min number of painters!

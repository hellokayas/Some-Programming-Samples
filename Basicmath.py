#Given an even number ( greater than 2 ), return two prime numbers whose sum will be equal to given number.

import math
#check if a number is prime
def isprime(n):
	if n == 1:
		return False
	for i in range(2,int(math.sqrt(n))+1):
		if n%i == 0:
			return False
			break
	return True

#the main fucntion
def primesum(n):
	for i in range(2, n//2 + 1):
		if isprime(i) and isprime(n-i):
			return [i,n-i]
	return [0,0]

#f there are more than one solutions possible, this returns the lexicographically smaller solution.

# Now this next function checks if a given number A is perfect power of some positive integer.

def ispower(A):
	if A == 1:
		return True
	for x in range(2,int(math.sqrt(A))+1):
		p = x
		while (p <=A):
			p = p*x
			if p == A:
				return True
	return False

#the following function checks if given an  positive integer it is a palindrome or not

def ispalindrome(n):
	if int(str(n)[::-1]) == n:
		return True
	return False

# the following function takes an integer and reverses it keeping the sign unchanged.
def revint(A):
	if A == 0:
		return 0
	sgn = 0
	if A>0:
		sgn = 1
	else:
		sgn = -1
	result = int(str(abs(A))[::-1])
	if result > 2**31 - 1:
		return 0
	return sgn*result

#Rearrange a given array so that Arr[i] becomes Arr[Arr[i]] with O(1) extra space. All elements in the array are in the range [0, N-1]
#N * N does not overflow for a signed integer.

def rearrange(A):
	n = len(A)
	for i in range(n):
		A[i] += (A[A[i]] % n)*n
	for i in range(n):
		A[i] = A[i] // n

#the next function takes a string which is the title of excel sheet column and outputs the it cardinal number

def strtonum(s):
	result = 0
	for i in range(len(s)):
		result = result*26 + (ord(s[i]) - ord('A') + 1)
	return result

#the next function takes a number which is less than 10**10 and outputs a string which is suppose to be the title of excel column.

def numtostr(n):
	s = ""
	while (n!= 0):
		n = n - 1
		temp = n%26
		n = n//26
		s = s + chr(temp + ord('A'))
	return s[::-1]

# the following function returns the number of trailing zeroes at the end of factorial of a given number.

def trailzero(n):
	ans = 0
	while(n!=0):
		ans = ans + n//5
		ans = ans//5
	return ans

#A robot is located at the top-left corner of an A x B grid.The robot can only move either down or right at any point in time and reach bottom right corner.
#How many possible unique paths are there?

def uniquepaths(A,B):
	soln = [[0 for i in range(B)] for j in range(A)]
	for i in range(A):
		soln[i][0] = 1# we fillup the top row and the left most column with the initial values
	for i in range(B):# so that we can fillup the array with these values step by step using dp
		soln[0][i] = 1
	for i in range(1,A):
		for j in range(1,B):
			soln[i][j] = soln[i-1][j] + soln[i][j-1]#the dp step
	return soln[A-1][B-1]

#Given an integer array of n integers, find sum of bit differences in all pairs that can be formed from array elements. Bit difference of
# a pair (x, y) is count of different bits at same positions in binary representations of x and y.

def hammingdist(A):
	res = 0
	n = len(A)
	#The idea is to count differences at individual bit positions. We traverse from 0 to 31 and count numbers with i’th bit set.
	for i in range(32):
		# count number of elements with i'th bit set
		count = 0
		# We traverse from 0 to 31 and count numbers with i’th bit set. Let this count be ‘count’. There would be “n-count” numbers with
		# i’th bit not set.
		for j in range(n):
			if A[j] & (1<<i):# (1<<i) gives the number with 1 in the ith place followed by zeroes, this is the left shift operator.
				count += 1 
		res += count * (n-count) * 2
		#the reason for this formula is as every pair having one element which has set bit at i’th position and second element having
		# unset bit at i’th position contributes exactly 1 to sum, therefore total permutation count will be count*(n-count) and multiply
		# by 2 is due to one more repetition of all this type of pair as per given condition for making pair 1<=i,j<=N.
	return res

#Given a string find its lexicographic rank among its all permutations

def findrank(s):
	n = len(s)
	res = 0
	for i in range(n):
		rank = 1 #when we have found how many strings are there before s, the rank of s will be that + 1, so start with that increment
		for j in range(i+1,n):
			if s[i] > s[j]:#for each char in s,check how many char appearing after s should be before s lexicographically 
				rank += 1#increase the rank by that many count
		res = res + rank * factorial(n-i-1)#finding how many permutations are there with that s[i] in front of all such s[j]'s
	return res

# Find the nth Fibonacci number in the best possible time and space limit

def Fib(n):
	mod = 10**9 + 7
	if n == 0:
		return 0
	M = [[1,1],[1,0]]
	power(M,n-1)
	return M[0][0]

	def power(F,m):
		if m == 1:
			return F
		if m % 2 == 0:
			s = power(F, n//2)
			return multiply(s,s)
		else:
			s = power(F, n // 2)
			return multiply(F,M)

	def multiply(f,m):
		a = (f[0][0] * m[0][0] % mod + f[0][1] * m[1][0] % mod) % mod
    	b = (f[0][0] * m[0][1] % mod + f[0][1] * m[1][1] % mod) % mod
    	c = (f[1][0] * m[0][0] % mod + f[1][1] * m[1][0] % mod) % mod
    	d = (f[1][0] * m[0][1] % mod + f[1][1] * m[1][1] % mod) % mod
    	f[0][0] = a
    	f[0][1] = b
    	f[1][0] = c
    	f[1][1] = d
    	return f

#Given a number find the next number greater than this having the same set of digits.The num is given as string.

def nextnum(s):
	n = len(s)
	for i in range(n-2,-1,-1): #we start removing parts of s from the back
		tail = sorted(s[i:])#we sort this removed part
		for j in len(tail):#then we take one number from this sorted part and insert it in the last position to find the next num
			m = int(s[:i] + tail[j] + "".join(tail[:j] + tail[j+1:]))#we do one num from the sorted part at a time
			if m > int(s):#check if the formed number crosses s
				return m
	return "-1"#the given num is the lasrgest possible with those digits and complexity is O(n^2 logn)
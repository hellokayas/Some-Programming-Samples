import math
import sys

# | is bitwise or
# & and ~ are bitwise and and not respectively.
# x << y = x * 2**y which is x has its bits shifted to right by y places(think like multiplying by power of 10)
# x>>y = x // 2**y  left shift 
# can test if two bits equal or not by xoring them(^). 
# Can flip a bit by xoring it with 1

#Write a function that takes an unsigned integer and returns the number of 1 bits it has.

#Iterate 32 times, each time determining if the ith bit is a ’1′ or not.
#This is probably the easiest solution, and the interviewer would probably not be too happy about it.
#MAIN IDEA : x - 1 would find the first set bit from the end, and then set it to 0, and set all the bits following it
#Which means if x = 10101001010100, Then x - 1 becomes 10101001010(011). All other bits in x - 1 remain unaffected.
#This means that if we do (x & (x - 1)), it would just unset the last set bit in x (which is why x&(x-1) is 0 for powers of 2).
# This method with x ^ (x-1) will also help finding number of trailing zeroes by setting them all to 1
def num_of_bit(A):
	count = 0
	while(A!= 0):
		A = A & (A-1)
		count += 1
	return count

# reverse a binary number(atmost 32 bit) bitwise

def reverse(A):
	ans = 0
	for i in range(32):
		if A & (1<<i):# 1<<i is 1 followed by i zeroes. A & (1<<i) gives the ith bit of A, if that is 1, then flip the just opposite bit 
			ans = ans | (1<<(31-i))#in ans, if the bit is 0, then we will have the opposite bit set in ans. 31-i gives the pos of the ith
	return ans# bit reading from left to right

#We define f(X, Y) as number of different corresponding bits in binary representation of X and Y. For example, f(2, 7) = 2, 
#since binary representation of 2 and 7 are 010 and 111, respectively. The first and the third bit differ, so f(2, 7) = 2
#given an array of N positive integers, A1, A2 ,…, AN. Find sum of f(Ai, Aj) for all pairs (i, j) such that 1 ≤ i, j ≤ N

def countbits(A):# we fix a bit and check how many a in A have that ith bit set and how many has ith bit unset
	n = len(A)# count = how many has ith bit set. So on the ith bit we have count*(N-count) differences already. Need take x2 since 
	ans = 0# f(a,b) = f(b,a). 
	for i in range(31):# So now iterate over 31 bits and find the difference based on each bit and add that to the ans
		count = 0# number of elems with ith bit set
		for a in A:
			if a & (1<<i):# gives the ith bit of a
				count += 1# if ith bit set increase count
		ans = ans + count*(n-count)*2
	return ans

#Given an integer array A of N integers, find the pair of integers in the array which have minimum XOR value. Report the minimum XOR value.
#Soln: Let’s suppose that the answer is not X[i] XOR X[i+1], but A XOR B and there exists C in the array such as A <= C <= B
#Next is the proof that either A XOR C or C XOR B are smaller than A XOR B.
#Let A[i] = 0/1 be the i-th bit in the binary representation of A
#Let B[i] = 0/1 be the i-th bit in the binary representation of B
#Let C[i] = 0/1 be the i-th bit in the binary representation of C
#This is with the assumption that all of A, B and C are padded with 0 on the left until they all have the same length
#Let i be the leftmost (biggest) index such that A[i] differs from B[i]. So B[i] = 1 and A[i] = 0. Two cases:
#1) C[i] = A[i] = 0, then (A XOR C)[i] = 0 and (A XOR B)[i] = 1. This implies (A XOR C) < (A XOR B)
#2) C[i] = B[i] = 1, then (B XOR C)[i] = 0 and (A XOR B)[i] = 1. This implies (B XOR C) < (A XOR B)

def minxor(A):#The first step is to sort the array. The answer will be the minimal value of X[i] XOR X[i+1] for every i
	n = len(A)
	A.sort()
	result = float("inf")
	for i in range(n):
		val = A[i] ^ A[i+1]
		result = min(result,val)
	return result

#Given an array of integers, every element appears thrice except for one which occurs once. Find that element which does not appear thrice.
#If O(1) space constraint was not there, you could've gone for a hashmap with values being the count of occurrences in O(n) time.
# Soln: Run a loop for all elements in array. At the end of every iteration, maintain following two values.

#ones: The bits that have appeared 1st time or 4th time or 7th time .. etc.
#twos: The bits that have appeared 2nd time or 5th time or 8th time .. etc.
#Finally, we return the value of ‘ones’

#How to maintain the values of ‘ones’ and ‘twos’?
#‘ones’ and ‘twos’ are initialized as 0. For every new element in array, find out the common set bits in the new element and previous
# value of ‘ones’. These common set bits are actually the bits that should be added to ‘twos’. So do bitwise OR of the common set bits 
#with ‘twos’. ‘twos’ also gets some extra bits that appear third time. These extra bits are removed later.
#Update ‘ones’ by doing XOR of new element with previous value of ‘ones’. There may be some bits which appear 3rd time. These extra bits
#are also removed later.

# Both ‘ones’ and ‘twos’ contain those extra bits which appear 3rd time. Remove these extra bits by finding out common set bits in
# ‘ones’ and ‘twos’.

def getSingle(arr, n): 
    ones = 0
    twos = 0
      
    for i in range(n): 
        # one & arr[i]" gives the bits that 
        # are there in both 'ones' and new 
        # element from arr[]. We add these 
        # bits to 'twos' using bitwise OR 
        twos = twos | (ones & arr[i]) 
          
        # one & arr[i]" gives the bits that 
        # are there in both 'ones' and new 
        # element from arr[]. We add these 
        # bits to 'twos' using bitwise OR 
        ones = ones ^ arr[i] 
          
        # The common bits are those bits  
        # which appear third time. So these 
        # bits should not be there in both  
        # 'ones' and 'twos'. common_bit_mask 
        # contains all these bits as 0, so 
        # that the bits can be removed from 
        # 'ones' and 'twos' 
        common_bit_mask = ~(ones & twos) 
          
        # Remove common bits (the bits that  
        # appear third time) from 'ones' 
        ones &= common_bit_mask 
          
        # Remove common bits (the bits that 
        # appear third time) from 'twos' 
        twos &= common_bit_mask 
    return ones 
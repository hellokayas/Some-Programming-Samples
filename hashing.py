import string
import math
import sys
from collections import Counter
from collections import defaultdict
'''
A number can be broken into different contiguous sub-subsequence parts.  Suppose, a number 3245 can be broken into parts
 like 3 2 4 5 32 24 45 324 245.  And this number is a COLORFUL number, since product of every digit of a contiguous
  subsequence is different. Given a num decide if it is colorful(1) or not(0)'''

def iscolorful(A):
	A = str(A)
	A = list(A)
	A = list(map(int,A)) # the above three lines make a num 21458 into [2,1,4,5,8]
	p = set()# the hash map
	for i in range(len(A)):
		prod = 1
		for j in range(i,len(A)):# running in O(n^2)
			prod *= A[j]
		if prod in p:# do each prod and chek if it is in the set
			return 0
		p.add(prod)
	return 1

# Given an arr, Find the largest continuous sequence in a array which sums to zero.
def lszero(A):
    sumdict = {0: -1}# this stores the cumulative sum of the values in A as keys and the num corresponding to the key is the index upto which
    # we have summed. if the dict keys never repeat that means no subseq sum to 0, it will repeat meaning the subseq intermediate to
    # the two repeating keys have the sum as 0
    maxlcs = 0# this is the len of the subseq which sums to 0, we update this if we find a longer such subseq summing to 0
    maxi,maxj = -1,-1# maxj is the faster index and the maxi is the one behind, they store the indices of the longest subseq summing 0
    total = 0# this is the cumulative sum of the val in A
    for i,val in enumerate(A):# the vals in A are numbered and tupled now starting the numbering from 1
        total += val
        if total in sumdict:# if found repeat, so found a subseq which sums to 0
            if maxlcs < i - sumdict[total]:# check if this is a longer subseq than the last one which summed to 0
                maxlcs = i - sumdict[total]# if so time to update!
                maxj = i# update
                maxi = sumdict[total] + 1# +1 is done so that the gap between the two indices = fast - slow -1 which should be the gap
        else:
            sumdict[total] = i# if not found, new total, add to the dict
    if maxlcs:
        return A[maxi:maxj + 1]# if atleast one subseq found summing to 0, uoutput the largest one, else return None, bydefault

#Given an array of integers, find two numbers such that they add up to a specific target number. If multiple solutions exist, output
# the one where index2 is minimum. If there are multiple solutions with the minimum index2, choose the one with minimum index1 out
# of them
def twosum(A,k):
    dict = {}
    for i in range(len(A)):
        if k-A[i] in dict:# also keep searching for the target-a[i] which should be the other summand, if found, done!
            return [dict[k-A[i]]+1, i+1]# just output the indices +1 since in A, indices start from 0
        elif A[i] not in dict:# the dict keeps track of the new values in A encountered
            dict[A[i]] = i# if not seen, enter the value with the key as the val
    return []

#Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets
# in the array which gives the sum of target. Elements in a quadruplet (a,b,c,d) must be in non-descending order. (ie, a ≤ b ≤ c ≤ d)
# The solution set must not contain duplicate quadruplets.
def foursum(A,k):
    A.sort()#When the array is sorted, try to fix the least and second least integer by looping over it.
    result = set()# to prevent duplicates in the result
    for i in range(len(A)):
        for j in range(i+1,len(A)):#Lets say the least integer in the solution is arr[i] and second least is arr[j]
            high = len(A)-1# lets try the 2 pointer approach. If we fix the two pointers at the end ( that is, j+1 and end of array ), we look at the sum.
#If the sum is smaller than the sum we want, we increase the first pointer to increase the sum.
#If the sum is bigger than the sum we want, we decrease the end pointer to reduce the sum.
            low = j+1
            while(low < high):
                x = A[i] + A[j] + A[low] + A[high]
                if x == k:
                    result.add((A[i],A[j],A[low],A[high]))# found a soln
                    high -= 1
                    low += 1
                elif x > k:
                    high -= 1
                else:
                    low += 1
    return sorted(result)# finally the result should be sorted

# Determine if a Sudoku is valid, The Sudoku board could be partially filled, where empty cells are filled with the character ‘.’
#A valid Sudoku board (partially filled) is not necessarily solvable. Only the filled cells need to be validated, we are not checking
# if the sudoku is solvable or not. The solution is nothing but checking all rows and all columns and all 3x3 matrices(the reqd ones) for
# repeating  numbers, if they are all distinct then the sudoku is valid. The input is a tuple of strings.

def isvalid(A):
	r = [[False]*9 for i in range(9)]# the rows
	c = [[False]*9 for i in range(9)]# the columns
	s = [[[False]*9 for i in range(3)] for i in range(3)]# the reqd 3x3 boxes

	for in range(9):
		for j in range(9):
			if A[i][j] != '.':# if it is  a digit
				num = ord(A[i][j]) - ord('1')# convert it to a number from str
				if r[i][num] or c[num][j] or s[i//3][j//3][num]:# if one of them is true, then repeating is happening since we had all of them
				# false to start with and we are making them true only if we have already put a value in that cell
					return 0# so repeatition found
				r[i][num] = True# found an elem in that cell
				c[num][j] = True# found an elem in that cell
				s[i//3][j//3][num] = True# found an elem in that cell. i//3 is happenning, look at the structure of s and how i should run 

	return 1# no repeatitions found

# Given an array A of integers and another non negative integer k, find if there exists 2 indices i and j such
# that A[i] - A[j] = k, i != j

def diff(A,k):
	complement = set()
	for a in A:
		if a-k in complement or a +k in complement:
			return 1
		complement.add(a)
	return 0

# Given an array of strings, return all groups of strings that are anagrams. Represent a group by a list of integers representing
# the index in the original list. Input : cat dog god tca       Output : [[1, 4], [2, 3]]

def anagrams(A):#Anagrams will map to the same string if the characters in the string are sorted.
#We can maintain a hashmap with the key being the sorted string and the value being the list of strings 
#( which have the sorted characters as key )
	dict = {}
	for i in range(len(A)):
		if "".join(sorted(A[i])) not in dict:
			dict["".join(sorted(A[i]))] = [i+1]
		else:
			dict["".join(sorted(A[i]))].append(i+1)
	return list(dict.values())

#Given an array A of integers, find the index of values that satisfy A + B = C + D, where A,B,C & D are integers values in the array
'''1) Return the indices `A1 B1 C1 D1`, so that 
  A[A1] + A[B1] = A[C1] + A[D1]
  A1 < B1, C1 < D1
  A1 < C1, B1 != D1, B1 != C1 

2) If there are more than one solutions, 
   then return the tuple of values which are lexicographical smallest. 

Assume we have two solutions
S1 : A1 B1 C1 D1 ( these are values of indices int the array )  
S2 : A2 B2 C2 D2

S1 is lexicographically smaller than S2 iff
  A1 < A2 OR
  A1 = A2 AND B1 < B2 OR
  A1 = A2 AND B1 = B2 AND C1 < C2 OR 
  A1 = A2 AND B1 = B2 AND C1 = C2 AND D1 < D2'''

def equal(A):
	seen = dict()
	result = []#Loop i = 1 to N :
    #Loop j = i + 1 to N :
     #   calculate sum
      #  If in hash table any index already exist for sum then 
       #     try to find out that it is valid solution or not IF Yes Then update solution
        #update hash table
    #EndLoop;
#EndLoop;
	for i in range(len(A)):
		for j in range(i+1, len(A)):
			currsum = A[i] + A[j]
			if currsum in seen:
				if i > seen[currsum][0] and seen[currsum][1] noti in (i,j):# here we are checking the conditions mentioned above
					result.append([seen[currsum][0],seen[currsum][1], A[i], A[j]])
				else:
					seen[currsum] = (i,j)
	result.sort()# due to returning the lexicographically the smallest result
	return result[0]

#Given a string, find the length of the longest substring without repeating characters
#The longest substring without repeating letters for "abcabcbb" is "abc", which the length is 3.
#For "bbbbb" the longest substring is "b", with the length of 1

def longestsubstr(s):
	maxlen = start = 0
	repeatchar = dict()
	for i in range(len(s)):
		if s[i] in repeatchar and start <= repeatchar[s[i]]:# this is the only case when we do not update the maxlen since, this is not
		# a valid substr, update the start
			start = repeatchar[s[i]] + 1# in all other cases, always update the maxlen
		else:
			maxlen = max(maxlen,i-start+1)
		repeatchar[s[i]] = i# whatever we do, always update the dict
	return maxlen

#You are given a string, S, and a list of words, L, that are all of the same length.
#Find all starting indices of substring(s) in S that is a concatenation of each word in L exactly once and without any
# intervening characters

def findsubstr(A,B):# @param A : string
    # @param B : tuple of strings
	m = len(A)
	wsize = len(B[0])
	lsize = len(B)
	h1 = counter(B)# counter is acting as the hash map here, if not allowed to use counter , will have to implement it first
	ans = []
	'''Lets say the size of every word is wsize and number of words is lsize.
You start at every index i. Look at every lsize number of chunks of size wsize and note down the words. Then match the set of words
 encountered to the set of words expected.

Now, lets look at ways we can optimize this.
Right now, to match words, we do it letter by letter. How about hashing the words ?
With hashing, hash(w1) + hash(w2) = hash(w2) + hash(w1).
In short, when adding the hashes, the order of words does not matter.
Can we optimize the matching of all the words encountered using that ? Can we use sliding pointers to move to index i + wsize from i ?'''
	for i in range(m-wsize*lsize+1):
		s = []
		for j in range(i+1,wsize*lsize+1):
			s.append(A[j:j+lsize])
		h2 = counter(s)
		if h1 == h2:
			ans.append(i)
	return ans

#Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
#If the fractional part is repeating, enclose the repeating part in parentheses.

def frac_to_decimal(A,B):
	'''When does the fractional part repeat ?
Hint: It has something to do with powers of 2 and 5 in denominator of reduced fraction.

You can extract floor part after division making numerator < denominator always.
Now can you relate this to elementary division algorithm for evaluating n/d when n is less than d.'''
	answer = []
	if (A<0) ^ (B<0) and (A!= 0):
		answer.append("-")
	A = abs(A)
	B = abs(B)
	'''Lets simulate the process of converting fraction to decimal. Lets look at the part where we have already figured out the integer part which is floor(numerator / denominator).
Now you are left with ( remainder = numerator%denominator ) / denominator.
If you remember the process of converting to decimal, at each step you do the following :

1) multiply the remainder by 10,
2) append remainder / denominator to your decimals
3) remainder = remainder % denominator.

At any moment, if your remainder becomes 0, you are done.

However, there is a problem with recurring decimals. For example if you look at 1/3, the remainder never becomes 0.

Notice one more important thing.
If you start with remainder = R at any point with denominator d, you will always get the same sequence of digits.
So, if your remainder repeats at any point of time, you know that the digits between the last occurrence of R will keep repeating.'''
	answer.append(str(A//B))
	if A % B:
		answer.append(".")
	rem = A % B
	hashmap = {}
	while rem and rem not in hashmap:
		hashmap[rem] = len(answer) - 1
		rem = rem * 10
		answer.append(str(rem//B))
		rem = rem % B
	if rem:# now we have to keep the repeating part within brackets as reqd
		answer.insert(hashmap[rem]+1, "(")# inserting the open bracket in the first occurence of the repeating digit
		answer.append(")")# during the 2nd occurence we have already left the while loop, so it is not there. just attach ) at the end
	return "".join(val for val in answer)# converting thr ans to a str

#Given N point on a 2D plane as pair of (x, y) co-ordinates, we need to find maximum number of point which lie on the same line.
'''Input : points[] = {-1, 1}, {0, 0}, {1, 1}, 
                    {2, 2}, {3, 3}, {3, 4} 
Output : 4
Then maximum number of point which lie on same
line are 4, those point are {0, 0}, {1, 1}, {2, 2},
{3, 3}'''

def maxpoints(A):# the arr containing the points

''' For each point p, calculate its slope with other points and use a map to record how many points have same slope, by which we can
 find out how many points are on same line with p as their one point. For each point keep doing the same thing and update the maximum
  number of point count found so far.

Some things to note in implementation are:
1) if two point are (x1, y1) and (x2, y2) then their slope will be (y2 – y1) / (x2 – x1) which can be a double value and can cause
 precision problems. To get rid of the precision problems, we treat slope as pair ((y2 – y1), (x2 – x1)) instead of ratio and reduce 
 pair by their gcd before inserting into map. In below code points which are vertical or repeated are treated separately.'''
	n = len(A)
	if n < 3:## upto two points all points will be part of the line
		return n
	answer = 0
	for i in A:# looping for each point 
		d = {}# Creating a dictionary for every new 
            # point to save memory 
		curr_max = 0
		duplicates = 0# number of dups
		for j in A:# this is what makes it O(n^2)
			if i != j:
				if i[0] == j[0]:# the case when we have found two points in the same vertical line with same x coordinates
					slope = "inf"
				else: slope = float(j[1]-i[1])/float(j[0]-i[0])# when non vertical, we cal the slopes usual way
				d[slope] = d.get(slope,0) + 1# the hashing step
				curr_max = max(curr_max,d[slope])# updating the part without duplicates
			else: duplicates += 1# thr case when i == j, updating it separately
		answer = max(answer,curr_max + duplicates)# updating the total, we do it for each point as we run on each point
	return answer

#Find the smallest window in a string containing all characters of another string.
#Input: string = “this is a test string”, pattern = “tist”
#Output: Minimum window is “t stri”
#Explanation: “t stri” contains all the characters of pattern.

def minwindow(string,pat):
	num_of_char = 256
	len1 = len(string)
	len2 = len(pat)
	if len1 < len2:# check if string's length is less than pattern's  
    # length. If yes then no such window can exist
		return ""
	hash_pat = [0]*num_of_char
	hash_str = [0]*num_of_char
	for i in range(len2):# store occurrence ofs characters of pattern 
		hash_pat[ord(pat[i])] += 1
	start, start_index = 0,-1# the start_index stores the start var from which we will start minimizing the window len next time
	minlen = float("inf")
	count = 0
	for j in range(len1):# start traversing the string 
		hash_str[ord(string[i])] += 1# count of characters
		if hash_pat[ord(string[j])] != 0 and hash_str[ord(string[j])] <= hash_pat[ord(string[j])]:
			count += 1# If string's char matches with  
        # pattern's char then increment count
		if count == len2:# if all the characters are matched 
			while(hash_pat[ord(string[start])] == 0 or hash_pat[ord(string[start])] <= hash_str[ord(string[start])]):
				if hash_pat[ord(string[start])] < hash_str[ord(string[start])]:# Try to minimize the window i.e., check if  
            # any character is occurring more no. of times  
            # than its occurrence in pattern, if yes  
            # then remove it from starting and also remove  
            # the useless characters
					hash_str[ord(string[start])] -= 1
				start += 1
			curr_len = j - start +1# update window size 
			if minlen > curr_len:
				minlen = curr_len
				start_index = start
	if start_index == -1:# If no window found 
		return ""
	return string[start: start + minlen] 

#Given an integer array A of size N containing 0's and 1's only.
#You need to find the length of the longest subarray having count of 1’s one more than count of 0’s.

'''
-> Consider all the 0’s in the array as ‘-1’.
-> Initialize sum = 0 and maxLen = 0.
-> Create a hash table having (sum, index) tuples.
-> For i = 0 to n-1, perform the following steps:
    -> If arr[i] is ‘0’ accumulate ‘-1’ to sum else accumulate ‘1’ to sum.
    -> If sum == 1, update maxLen = i+1.
    -> Else check whether sum is present in the hash table or not. 
       If not present, then add it to the hash table as (sum, i) pair.
    -> Check if (sum-1) is present in the hash table or not. 
       If present, then obtain index of (sum-1) from the hash table as index.
       Now check if maxLen is less than (i-index), then update maxLen = (i-index).
-> Return maxLen.'''

def solve(self, A):
        for i in range(len(A)):
            if A[i]==0:
                A[i]=-1
    
        if 1 not in A:
            return 0
        ans=1
        su=0
        d={}
        for i in range(len(A)):
            su+=A[i]
            if su==1:
                ans=max(ans,i+1)
                continue
            if su-1 in d:
                ans=max(ans,i-d[su-1])
            if su not in d:
                d[su]=i
            
        return ans
import math
import sys
import string
from itertools import groupby

#Given a string A. Return the string A after reversing the string word by word.
#"the sky is blue" --> "blue is sky the"

def revword(S):# join returns words joined by space. Strip removes irrelevant spaces from start and end of the str
	return "".join(S.strip().split()[::-1])# split decomposes the string into words by space, after that we reverse it and then join

#Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
#If the last word does not exist, return 0

def lastwordlen(s):
	wordlength = 0
	wordfound = False
	new_s = s[::-1]# now look for the first word after spaces
	for c in new_s:
		if c == " ":# if space, then not found first word
			if wordfound == True:# if found first word and c == " ", that means this is 2nd word, so return wordlength accumulated
				return wordlength
			continue# if first word not found and getting spaces only, go on
		wordlength += 1# first letter encounter after c first gets a non space character
		wordfound = True# c = a char not space, then set flag to true
	return wordlength# if nothing found this returns 0 at the end

#Given the array of strings A, you need to find the longest string S which is the prefix of ALL the strings in the array

def longcomprefix(A):
	min_word = min(A,key = lambda word : len(word))# find the word in A which has the least length, then the result will be its substring
	m = len(A)
	n = len(min_word)# the algorithm is O(mn)
	for i in range(n):# fix the first letter of the smallest word
		for j in range(m):# check the first letter of all the strings, if matches, then start with 2nd letter of min_word and match
			if A[j][i] != min_word[i]:# if no match, then upto this is the longest match got, if fails for any j, then fails totally
				return min_word[:i]
	return min_word# all matches done and everything has gone through, so the whole of smallest string is prefix of all the rest strings

#1 is read off as one 1 or 11. 11 is read off as two 1s or 21. 21 is read off as one 2, then one 1 or 1211.
#Given an integer A, generate the Ath number in the sequence

def countnsay(A):
	if A == 1:
		return "1"
	seed = countnsay(A-1)# using recursion
	ans = ""
	for i,n in groupby(seed):# look at https://www.geeksforgeeks.org/itertools-groupby-in-python/
		ans += str(len(list(n))) + i[0]
	return ans

#Compare two version numbers version1 and version2. If version1 > version2 return 1, If version1 < version2 return -1, otherwise return 0.
#For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level
# revision. 0.1 < 1.1 < 1.2 < 1.13 < 1.13.4 is a correct ordering of versions.

def compareversion(A,B):
	verA = list(map(int,A.split(".")))# split both the numbers by "." to compare the real numbers and put them in the lists verA and verB
	verB = list(map(int,B.split(".")))
	a = len(verA)
	b = len(verB)
	for index in range(max(a,b)):# iterate over maxlen so that we cover both the lists
		if index < a:# if index in range
			anum = verA[index]# temp variable anum reqd for comparison
		else:# index out of range, assign 0, bnum will be greater anyway if the value is nonzero
			anum = 0
		if index < b:
			bnum = verB[index]# again temp var for verB
		else:
			bnum = 0
		if anum > bnum :# we compare here
			return 1
		if bnum > anum:
			return -1
	return 0# equlaity has hold throughout

# string to integer atoi function to be implemented

def atoi(s):
        s = s.strip() # strips all spaces on left and right
        if not s: return 0
        sign = -1 if s[0] == '-' else 1
        val, index = 0, 0
        if s[0] in ['+', '-']: index = 1
        while index < len(s) and s[index].isdigit():
            val = val*10 + ord(s[index]) - ord('0') # assumes there're no invalid chars in given string
            index += 1
        #return sign*val
        return max(-2**31, min(sign * val,2**31-1))

# convert roman(given as string) to Integer

def romanToInt(A):
#The key is to notice that in a valid Roman numeral representation the letter with the most value always occurs at the start of the string.
#Whenever a letter with lesser value precedes a letter of higher value, it means its value has to be added as negative of that letter’s
#value. In all other cases, the values get added.
        
        bank = {'X': 10, 'V' : 5, 'I' : 1, 'L' : 50, 'C' : 100, 'D' : 500, 'M' : 1000}
        
        res = 0
        for i in xrange(0, len(A)):
            cur = bank[A[i]]
            if i+1 < len(A) and cur < bank[A[i+1]]:
                res -= cur
            else:
                res+= cur
                
        return res

def intToRoman(num):
        num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

        roman = ''
    
        while num > 0:
            for i, r in num_map:
                while num >= i:
                    roman += r
                    num -= i
    
        return roman

# given an int A, determine if it is a power of 2

def ispower(A):
        s=int(A)
        if s<=2:
            return 0
        if  s&(s-1)==0:# another way: if(math.ceil(math.log2(num))==math.log2(num)): return 1
            return 1
        return 0

# given two binary strings add them

def add_bin_strings(x,y):
	maxlen = max(len(x),len(y))# find the maxlen to pad the left of both strs with 0's
	x = x.zfill(maxlen)
	y = y.zfill(maxlen)
	carry = 0# initialize
	result = ""
	for i in range(maxlen-1,-1,-1):# going backwards
		temp = carry# staritng to add along each columns and carry
		temp += 1 if x[i] == "1" else 0
		temp += 1 if y[i] == "1" else 0# at this stage, temp is the sum of the current column along with carry
		result += "1" if temp % 2 == 1 else "0"# acc to rules of bin addition
		carry = 0 if temp < 2 else 1# acc rules of bin addition, temp = 1 means only one number in the column is 1
	if carry != 0:# if no carry, we have got the result
		result = "1" + result# else we add the most significant bit
	return result.zfill(maxlen)

# Given two integers as strings, multiply them without any lib funcs

def str_multiply(num1,num2):
	len1,len2 = len(num1),len(num2)
	num1 = list(map(int,reversed(num1)))# converting each string to list whose elems are the digits in the reverse order
	num2 = list(map(int,reversed(num2)))
	result = [0]*(len1+len2)# init a result arr with all zeroes
	for j in range(len2):
		for i in range(len1):
			result[i+j] += num1[i]*num2[j]
			result[i+j+1] += result[i+j] // 10# this is adding the carry
			result[i+j] = result[i+j] % 10# can have only one digit in each slot of the arr
	i = len(result) - 1# going from back and removing the extra zeroes(now on right but the result is reversed now)
	while(i>0 and result[i] == 0):
		i -= 1# find that i upto where it has been padded with 0s
	return "".join(map(str,result[:i+1][::-1]))# result[:i+1] removes the 0 padding and then [::-1] reverses, then converts to str

#Given an string A. The only operation allowed is to insert characters in the beginning of the string.
#Find how many minimum characters are needed to be inserted to make the string a palindrome string

def mincharpalin(A):# for the best soln see geeksforgeeks how to implement ideas from KMP algorithm
	if A[::-1] == A:# already palindrome
		return 0
	j = len(A)-1
	while(j>=0):# goinf back starting from full length
		B = A[:j]# removing char one by one from the back
		if B == B[::-1]:# if palindrome, then we have removed len(A)-j chars from back we have to append those in front to get 
			return len(A)-j# a palindrome
		j -= 1# else just go on iterating
	return len(A)-1# never a palindrome, the whole str needs to be appended in the front in reverse way to make a planindrome

# Given a string S, find the longest palindromic substring S[i...j] in S
# for the best soln refer: https://www.geeksforgeeks.org/manachers-algorithm-linear-time-longest-palindromic-substring-part-3-2/
def longestpalin(s):
	n = len(s)
	for width in range(n-1,0,-1):# start with the max possible width of the substr and reduce upto substr of form [i,i+1]
		index = 0# starting from the first so that if there is two substrs of equal length, we return the first one
		while (index + width < n):# keeping a fixed width here
			substr = s[index:index+width]
			if substr == substr[::-1]:# the palindrome check
				return substr# if found
			index += 1# else go to the next index keeping the fixed width

#Given a string A consisting only of lowercase characters, we need to check whether it is possible to make this string a palindrome
#after removing exactly one character from this. If it is possible then return 1 else return 0.

def solve(A):
        if(A==A[::-1]):# already palindrome
            return(1)
        i=0
        j=len(A)-1
        while(j>=i):# two pointer approach
            if(A[i]==A[j]):# while equal go on to find the first mismatch
                i+=1
                j-=1
                continue# can be ignored
            elif(A[i]!=A[j]):# when the 1st mismatch found, 
                str1=A[i+1:j+1]# remove A[i] and check if palindrome
                str2=A[i:j]# remove A[j] and check palindrome
                if(str1==str1[::-1] or str2==str2[::-1]):# checking is done here, if one of them is, then done!
                    return(1)
                else:
                    return(0)
            i+=1# can be ignored
            j-=1# can be ignored
        return(0)

#Given a string A of parantheses ‘(‘ or ‘)’. The task is to find minimum number of parentheses ‘(‘ or ‘)’ (at any positions) we must 
# add to make the resulting parentheses string valid.

def solve(A):
	count1 = 0# how many ")" will be reqd corresponding to the open br
	count2 = 0# how many "(" will be reqd corresponding to closed br
	for i in A:
		if i == "(":
			count1 += 1
		if i == ")" and not count1:# we putting close br but no open brackets there
			count2 += 1
		if i == ")" and count1> 0:# closing br but many open already present, since we can put anywhere the new brackets
			count1 -= 1# the problem is lot easier
	return count1+count2
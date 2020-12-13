import math
import sys
import string
from collections import Counter
from collections import defaultdict
from collections import deque
'''
There are N children standing in a line. Each child is assigned a rating value.

You are giving candies to these children subjected to the following requirements:

1. Each child must have at least one candy.
2. Children with a higher rating get more candies than their neighbors.
What is the minimum candies you must give?
Soln appraoch: Start with the guy with the least rating. Obviously he will receive 1 candy.
If he did recieve more than one candy, we could lower it to 1 as none of the neighbor have higher rating.
Now lets move to the one which is second least. If the least element is its neighbor, then it receives 2 candies, else we can get away
 with assigning it just one candy.
We keep repeating the same process to arrive at optimal solution.'''
def candy(A):
	if len(A) <= 1:
		return len(A)
	val = [1 for _ in range(len(A))]# everyone should get atleast one candy
	for i in range(1,len(A)):# in this pass check the neighbours on left side and increment as reqd
		if A[i] > A[i-1]:
			val[i] = max(val[i],val[i-1]+1)
	for i in range(len(A)-2,-1,-1):# in this reverse pass, check neighbors on the right side and increment as reqd
		if A[i] > A[i+1]:
			val[i] = max(val[i],val[i+1]+1)
	return sum(val)# sum is the total num of candies reqd

'''
There are N Mice and N holes are placed in a straight line.
Each hole can accomodate only 1 mouse.
A mouse can stay at his position, move one step right from x to x + 1, or move one step left from x to x − 1. Any of these moves consumes 1 minute.
Assign mice to holes so that the time when the last mouse gets inside a hole is minimized.

Example:

positions of mice are:
4 -4 2
positions of holes are:
4 0 5

Assign mouse at position x=4 to hole at position x=4 : Time taken is 0 minutes 
Assign mouse at position x=-4 to hole at position x=0 : Time taken is 4 minutes 
Assign mouse at position x=2 to hole at position x=5 : Time taken is 3 minutes 
After 4 minutes all of the mice are in the holes.

Since, there is no combination possible where the last mouse's time is less than 4, 
answer = 4.
Approach:
sort mice positions (in any order)
sort hole positions 

Loop i = 1 to N:
    update ans according to the value of |mice(i) - hole(i)|

Proof of correctness:

Let i1 < i2 be the positions of two mice and let j1 < j2 be the positions of two holes.
It suffices to show via case analysis that

max(|i1 - j1|, |i2 - j2|) <= max(|i1 - j2|, |i2 - j1|) , 
    where '|a - b|' represent absolute value of (a - b)
since it follows by induction that every assignment can be transformed by a series of swaps into the sorted assignment, where
 none of these swaps increases the makespan'''
def mice(self, a, b):
        return max(abs(i - j) for i, j in zip(sorted(a), sorted(b)))

'''
Given an array of size n, find the majority element. The majority element is the element that appears more than floor(n/2) times.

You may assume that the array is non-empty and the majority element always exist in the array.
Soln Approach: The majority element is the element that occurs more than half of the size of the array. This means that the majority
 element occurs more than all the other elements combined. That is, if you count the number of times the majority element appears,
  and subtract the number of occurrences of all the other elements, you will get a positive number.

So if you count the occurrences of some element, and subtract the number of occurrences of all other elements and get the number 0 - then
 your original element can't be a majority element. This is the basis for a correct algorithm:

Declare two variables, counter and possible_element. Iterate the stream, if the counter is 0 - your overwrite the possible element and
 initialize the counter, if the number is the same as possible element - increase the counter, otherwise decrease it.'''
def majorityElement(A):
	    counter,major = 0,None
	    for i in A:
	        if counter == 0:
	            counter,major = 1,i
	        elif i == major:
	            counter += 1
	        else:
	            counter -= 1
	    return major
'''
Given arrival and departure times of all trains that reach a railway station, the task is to find the minimum number of platforms required
 for the railway station so that no train waits.
We are given two arrays which represent arrival and departure times of trains that stop.
Algorithm:
Sort the arrival and departure time of trains.
Create two pointers i=0, and j=0 and a variable to store ans and current count plat
Run a loop while i<n and j<n and compare the ith element of arrival array and jth element of departure array.
if the arrival time is less than or equal to departure then one more platform is needed so increase the count, i.e. plat++ and increment i
Else if the arrival time greater than departure then one less platform is needed so decrease the count, i.e. plat++ and increment j
Update the ans, i.e ans = max(ans, plat).
Implementation: This doesn’t create a single sorted list of all events, rather it individually sorts arr[] and dep[] arrays, 
and then uses merge process of merge sort to process them together as a single sorted array.'''
def findPlatform(arr, dep, n): 
  
    # Sort arrival and 
    # departure arrays 
    arr.sort() 
    dep.sort() 
   
    # plat_needed indicates 
    # number of platforms 
    # needed at a time 
    plat_needed = 1
    result = 1
    i = 1
    j = 0
   
    # Similar to merge in 
    # merge sort to process  
    # all events in sorted order 
    while (i < n and j < n): 
     
        # If next event in sorted 
        # order is arrival,  
        # increment count of 
        # platforms needed 
        if (arr[i] <= dep[j]): 
          
            plat_needed+= 1
            i+= 1
          
   
        # Else decrement count 
        # of platforms needed 
        elif (arr[i] > dep[j]): 
          
            plat_needed-= 1
            j+= 1
  
        # Update result if needed  
        if (plat_needed > result):  
            result = plat_needed 
          
    return result 
'''
Given a set of N intervals denoted by 2D array A of size N x 2, the task is to find the length of maximal set of mutually disjoint
 intervals. Two intervals [x, y] & [p, q] are said to be disjoint if they do not have any point in common.
Return a integer denoting the length of maximal set of mutually disjoint intervals.
Soln:
-> Sort the intervals, with respect to their end points.
-> Now, traverse through all the intervals, if we get two overlapping intervals, then greedily choose the interval with lower end point since, choosing it will ensure that intervals further can be accommodated without any overlap.
-> Apply the same procedure for all the intervals and return the count of intervals which satisfy the above criteria.'''
def countinterval(A):
	def takesecond(elem):
		return elem[1]
	A = sorted(A,key=takesecond)# sort wrt the end pts of the intervals
	count,endpt = 1,A[0][1]# initialize the ans as 1
	for i in range(1,len(A)):
		if A[i][0] > endpt:# this means does not overlap
			count += 1
			endpt = A[i][1]# update the endpt
	return count
'''
Given two integer arrays A and B of size N.
There are N gas stations along a circular route, where the amount of gas at station i is A[i].

You have a car with an unlimited gas tank and it costs B[i] of gas to travel from station i
to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.

Return the minimum starting gas station’s index if you can travel around the circuit once, otherwise return -1.

You can only travel in one direction. i to i+1, i+2, … n-1, 0, 1, 2.. Completing the circuit means starting at i and
ending up at i again.
Soln:
The bruteforce solution should be obvious. Start from every i, and check to see if every point is reachable with the gas available.
 Return the first i for which you can complete the trip without the gas reaching a negative number.
This approach would however be quadratic.

Lets look at how we can improve.
1) If sum of gas is more than sum of cost, does it imply that there always is a solution ?
2) Lets say you start at i, and hit first negative of sum(gas) - sum(cost) at j. We know TotalSum(gas) - TotalSum(cost) > 0.
 What happens if you start at j + 1 instead ? Does it cover the validity clause for i to j already ?'''
def canCompleteCircuit(self, gas, cost):
        if sum(gas) < sum(cost):
            return -1
        res = 0
        cur_gas = 0
        for i in range(len(gas)):
            cur_gas += gas[i] - cost[i]
            if cur_gas < 0:
                cur_gas = 0
                res = i + 1
        return res
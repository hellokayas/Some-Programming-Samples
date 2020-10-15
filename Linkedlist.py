import sys
import math
import string

# A simple Python program to introduce a linked list 
  
# Node class 
class Node: 
  
    # Function to initialise the node object 
    def __init__(self, data): 
        self.data = data  # Assign data 
        self.next = None  # Initialize next as null 
  
  
# Linked List class contains a Node object 
class LinkedList: 
  
    # Function to initialize head 
    def __init__(self): 
        self.head = None
  
  
# Code execution starts here 
if __name__=='__main__': 
  
    # Start with the empty list 
    llist = LinkedList() # using the linkedlist class here to create a linkedlist
  
    llist.head = Node(1) # using the Node class here to create a node
    second = Node(2) 
    third = Node(3) 
  
    ''' 
    Three nodes have been created. 
    We have references to these three blocks as head, 
    second and third 
  
    llist.head        second              third 
         |                |                  | 
         |                |                  | 
    +----+------+     +----+------+     +----+------+ 
    | 1  | None |     | 2  | None |     |  3 | None | 
    +----+------+     +----+------+     +----+------+ 
    '''
  
    llist.head.next = second; # Link first node with second  
  
    ''' 
    Now next of first Node refers to second.  So they 
    both are linked. 
  
    llist.head        second              third 
         |                |                  | 
         |                |                  | 
    +----+------+     +----+------+     +----+------+ 
    | 1  |  o-------->| 2  | null |     |  3 | null | 
    +----+------+     +----+------+     +----+------+  
    '''
  
    second.next = third; # Link second node with the third node 
  
    ''' 
    Now next of second Node refers to third.  So all three 
    nodes are linked. 
  
    llist.head        second              third 
         |                |                  | 
         |                |                  | 
    +----+------+     +----+------+     +----+------+ 
    | 1  |  o-------->| 2  |  o-------->|  3 | null | 
    +----+------+     +----+------+     +----+------+  '''

# print a given linked list L

def printlist(L):
	temp = L.head
	while (temp):
		print(temp.data)
		temp = temp.next

# Given a linkedlist by giving the head node of the LL, reverse the LL and return the head node of the reversed LL
# 1->2->3->4->5->NULL  gives output 5->4->3->2->1->NULL

def reverseLL(L):
	prev = None
	curr = L.head
	while (curr):
		NEXT = curr.next #Before changing next of current, store next node
		curr.next = prev # Now change next of current This is where actual reversing happens, as the arrow gets turned back
		prev = curr #Move prev and curr one step forward
		curr = NEXT
	L.head = prev # finally putting the new head at the end
	# the function changes the LL in place and returns nothing. Anyway we can also return prev if needed.

# the following func adds a new node along with data at the front of the LL

def addnode(newdata):
	newnode = Node(newdata)
	newnode.next = head
	head = newnode

# find the length of a given LL

def length(L):# this means the head of LL is given
	ans = 0
	while(L):
		ans += 1
		L = L.next
	return ans

# the next func finds the node at which the intersection of two singly linked lists begins

def getintersection(A,B):# actually we are not looking for intersection but where the LLs merge permanently
	m = length(A)
	n = length(B)
	d = n-m
	if m >= n:# we want to keep the longer list first, so that we can traverse the extra length first
		d = m-n
		A,B = B,A
	c = 0
	while c < d:# traversing the extra length first since to find where the 2 LLs merge, the extra part does not count
		B = B.next
		c += 1
	while (A and B):# now both of them have equal length, so begin the search for the node where they merge
		if A == B:
			return A
		A = A.next
		B = B.next
	return None# they never merge

# Given a sorted LL, remove the duplicates and return the head

def removeduplicates(A):
	head = A# store the head which to be returned at the end after we change the LL inplace
	while A:# while not running to the null node
		while A.next and A.next.data == A.data:# while we are treading over the equal values which are clustered together in sorted LL
			A.next = A.next.next# we go over the LL till we find the new value where the mismatch happens in the while loop
		A = A.next# we set the arrow from the last A to the new value, which becomes our new A for the outer while loop
	return head

# Remove all occurrences of duplicates from a sorted Linked List
# 23->28->28->35->49->49->53->53 outputs  23->35
# 11->11->11->11->75->75 Outputs : empty List

'''The idea is to maintain a pointer (prev) to the node which just previous to the block of nodes we are checking for duplicates.
 In the first example, the pointer prev would point to 23 while we check for duplicates for the node 28. Once we reach the last duplicate
  node with value 28 (name it current pointer), we can make the next field of prev node to be the next of current and update 
  current=current.next. This would delete the block of nodes with value 28 which has duplicates.'''
def deleteduplicates(A):
	dummy = Node(0)# the new node before the head is created
	dummy.next = A# the dummy is connected to the head
	prev = dummy# the slower pointer starts at this new node
	while A:
		cur = A# the faster pointer starts at A
		while A.next and cur.data == A.next.data:# while we are finding duplicates
			A = A.next# go for the next duplicate
		if cur == A:# this happens if we have not found any duplicate, so the cur pointer still points to A, A has not moved
			prev = prev.next# move the slower pointer and update the new A anyway at line 162
		else:# A has moved and cur no longer points to A
			prev.next = A.next# prev should be connected to the new value found, A was the the last duplicate left, so connect to A.next
		A = A.next# anyway we have to update A to the next value
	return dummy.next# this is the head of LL after everything has been done in place

# given a LL convert it to arr and then back to LL

def LL_to_arr(A):
	arr = []
	while A:
		arr.append(A.data)
		A = A.next
	return arr
def arr_to_LL(arr):
	new = Node(arr[0])
	curr = new
	for i in range(1,len(arr)):
		curr.next = Node(arr[i])
		curr = curr.next
	return new 

# Given LL, delete Nth node from the end, if len(LL) < N, delete the head
#Soln:
#Take two pointers, first will point to the head of the linked list and second will point to the Nth node from the beginning.
#Now keep increment both the pointers by one at the same time until second is pointing to the last node of the linked list.
#After the operations from the previous step, first pointer should be pointing to the Nth node from the end by now.
# So, delete the node first pointer is pointing to.
def deleteN(A,n):# remove nth node from end
	first = A# this one at head
	second = A
	for i in range(n):# the 2nd one starts moving to stop at the nth node
		if second.next == None:# this happens then stop and check
			if i == n-1:# this is the case when we just remove the head of the list and return the new LL
				A = A.next# removing the head
			return A
		second = second.next# if no problem in 191 line, then 2nd pointer moves on
	while(second.next != None):# now the first pointer starts moving till the 2nd pointer runs out of the LL
		first = first.next
		second = second.next# this moving simultaneously, when runs out, means w have found the n-1 th node from end

	first.next = first.next.next# join the n-1 th node to the n+1 th node from the end
	return A# finally return the head

# Pairwise swap the adjacent nodes of LL starting from head
# 1->2->3->4->5->6->NULL outputs 2->1->4->3->6->5->NULL
# 1->2->3->4->5->NULL outputs 2->1->4->3->5->NULL

def pairwiseswap(A):
	new = Node(None)# this is created to keep track of the head after the val in head gets flipped
	new.next = A# it is linked to the head
	temp = A
	if not temp:# empty LL
		return
	while (temp and temp.next):# the next node should not be None for the flip to happen
		if temp.data == temp.next.data:# if equal val then no sense in flipping
			temp = temp.next.next
		else:# in this case we flip the val in the nodes
			temp.data,temp.next.data = temp.next.data,temp
			temp = temp.next.next# move on to the next swap
	return new.next# this points to the new head now

#Given a singly linked list, rotate the linked list counter-clockwise by k nodes. Where k is a given positive integer. 
#For example, if the given linked list is 10->20->30->40->50->60 and k is 4, the list should be modified to 50->60->10->20->30->40
# if k is more than number of nodes then return LL unchanged.

def rotate(A,k):
	if k == 0:
		return
	# Let us understand the below code for example k = 4 
        # and list = 10->20->30->40->50->60 
	curr = A
	count = 1
	# current will either point to kth or NULL after 
        # this loop 
        # current will point to node 40 in the above example 
	while(count < k and curr.next != None):
		curr = curr.next
		count += 1
		# If current is None, k is greater than or equal  
        # to count of nodes in linked list. Don't change 
        # the list in this case 
	if curr == None:
		return
		# current points to kth node. Store it in a variable 
        # kth node points to node 40 in the above example 
	kthnode = curr
	 # current will point to lsat node after this loop 
        # current will point to node 60 in above example
	while(curr.next != None):
		curr = curr.next
		# Change next of last node to previous head 
        # Next of 60 is now changed to node 10 
	curr.next = A
	head = kthnode.next
	# Change head to (k + 1)th node 
        # head is not changed to node 50
	kthnode.next = None
	# change next of kth node to NULL  
        # next of 40 is not NULL 
	return head

#Given a singly linked list and an integer K, reverses the nodes of the list K at a time and returns modified linked list.

def kreverse(A,k):
	curr = A
	prev = None
	after = None
	count = 0

	while(count < k and curr.next != None):# # Reverse first k nodes of the linked list
		after = curr.next
		curr.next = prev
		prev = curr
		curr = after
		count += 1
		# next is now a pointer to (k+1)th node 
        # recursively call for the list starting 
        # from current. And make rest of the list as 
        # next of first node 
	if after != None:
		A.next = kreverse(after,k)
	return prev# prev is new head of the input list

def partition(A,x):
	h = high = Node(None)# creating artificial nodes to keep track later
	l = low = Node(None)
	while(A):
		if A.data >= x:# maintain two diff LL for higher values and lower values than x
			h.next = Node(A.data)
			h = h.next# always maintain h as the last value in the list of greater values
		if A.data < x:# the exact same thing for values smaller than x
			l.next = Node(A.data)
			l = l.next
		A = A.next# A moves on till the end
	l.next = high.next# joining the last value of smaller LL to the 2nd val of the higher LL since the 1st val is None artificially made
	return low.next# return the head which is the 2nd val of the lower LL, since first val is the artificially created node in line 285

# Given two sorted LL, merge them

def mergesorted(A,B):
	i,j = A,B
	first,last = None,None
	while (i and j):
		if i.data < j.data:
			minnode = i
			i = i.next
		else:
			minnode = j
			j.next
		if last == None:
			first = last = minnode# initialize the first node which will be the head and returned at the end
		else:
			last.next = minnode# we build from the back
			last = minnode
	while i:# after one of A and B is exhausted, if A remains, this loop goes on
		last.next = i
		last = i
		i = i.next
	while j:# this loop goes on if A is shorter and so B goes on
		last.next = j
		last = j
		j = j.next
	last.next = None
	return first

# Now implement mergesort of a given LL

def mergesort(A):
	if A is None or A.next is None:
		return
	n = length(A)
	mid = A
	for i in range(n//2 - 1):
		mid = mid.next
	first = A# taking the head of the first list
	second = mid.next# the head of the 2nd list
	mid.next = None# cutting the original list from the middle
	return mergesorted(mergesort(first),mergesort(second))

# Given a node and a sorted LL(the head is given), insert the node in the LL in the right place

def insertsorted(node,sorted_list):
	if not sorted_list or node.data <= sorted_list.data:
		node.next = sorted_list
		sorted_list = node
	else:
		curr = sorted_list
		while (curr.next != None and curr.next.data < node.data):# find where the node actually fits
			curr = curr.next
		node.next = curr.next
		curr.next = node
	return sorted_list

# Given a LL, do an insertion sort with the insertsorted function

def insertionsortLL(A):
	if not A.next:
		return
	curr = A
	sorted_list = None
	while curr.next is not None:
		after = curr.next
		sorted_list = insertsorted(curr,sorted_list)# for every value we see, we run the above function once, which makes it O(n^2)
		curr = after
	return sorted_list

'''
given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes contain
 a single digit. Add the two numbers and return it as a linked list.

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8

    342 + 465 = 807
'''
def addlinkedlists(A,B):
	head = Node(None)
	curr = head
	carry = 0
	while(A and B):
		digitsum = carry + A.data + B.data
		carry = digitsum // 10
		val = digitsum % 10
		curr.next = Node(val)
		curr = curr.next
		A = A.next
		B = B.next
	while(A):# if B has shorter length and has ended, but A still continues, just insert the values remaining
		digitsum = carry + A.data
		carry = digitsum // 10
		val = digitsum % 10
		curr.next = Node(val)
		curr = curr.next
		A = A.next
	while(B):# when A has shorter length
		digitsum = carry + B.data
		carry = digitsum // 10
		val = digitsum % 10
		curr.next = Node(val)
		curr = curr.next
		B = B.next
	if carry != 0:# if carry is there, it means a new digit is to be added, so make a new node and put the curr -> new node
		curr.next = Node(carry)
	return head.next# head was none, our created node, actual head starts from next to head

# detecting loop in a LL using hash is easy! But this has O(n) space complexity. We have to return the first node where the loop starts
def detectCycle(A):
        s = set()
        temp = A
        while(temp):
            if temp in s:
                return temp
            s.add(temp)
            temp = temp.next
        return None
# the following with have O(1) space with same as above O(n) time by Floyd’s Cycle-Finding Algorithm
'''
List has no cycle:
The fast pointer reaches the end first and the run time depends on the list's length, which is O(n)O(n).

List has a cycle:
We break down the movement of the slow pointer into two steps, the non-cyclic part and the cyclic part:

The slow pointer takes "non-cyclic length" steps to enter the cycle. At this point, the fast pointer has already reached the cycle.
\text{Number of iterations} = \text{non-cyclic length} = NNumber of iterations=non-cyclic length=N

Both pointers are now in the cycle. Consider two runners running in a cycle - the fast runner moves 2 steps while the slow runner moves
 1 steps at a time. Since the speed difference is 1, it takes (distance between the 2 runners) / (difference of speed)
loops for the fast runner to catch up with the slow runner.
Therefore, the worst case time complexity is O(N+K) Space complexity : O(1)
look at https://www.geeksforgeeks.org/find-first-node-of-loop-in-a-linked-list/  for complete details of how this method is working
'''
def first_node_ofcycle(A):
	if not A: return None
	slow_p = A
	fast_p = A
	while(slow and fast and fast.next):
		slow = slow.next
		fast = fast.next.next
		if fast == slow: break
	if fast != slow:
		return None
	slow = A
	while(slow != fast):# why will they meet and when they will meet why will that point be the first node?? The ans is in the link
		slow = slow.next
		fast = fast.next
	return slow

# sort a LL of only 0 and 1
def solve(self, A):
        h=A
        
        a=0# counts num of zeroes in LL
        b=0# counts the num of ones in LL
        
        while h:# now count the zeroes and 1s using and store in a and b resp
            if h.val==0:
                a+=1
            else:
                b+=1
            h=h.next    
            
        h=A# now update the LL on the 2nd pass
        c=0
        while h:
            if c<a:# first put all the zeroes, a many of them are there
                c+=1
                h.val=0
            else:# when all the 0s are exhausted, now start writing the ones
                h.val=1
            h=h.next    
        return A

# Given L: L0 → L1 → … → Ln-1 → Ln reorder it to L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → …  and should be done inplace
# Given {1,2,3,4}, reorder it to {1,4,2,3}
'''
Soln: 1) Break the list from middle into 2 lists.
2) Reverse the latter half of the list.
3) Now merge the lists so that the nodes alternate.
'''

def reorder(A):
	head = A
	n = length(A)
	mid = A
	for i in range(n//2):
		mid = mid.next
	B = reverseLL(mid.next)
	''' cur=A   this is how reversing can be done for thi problem, though reverseLL is written above
        pre=None
        while cur!=None:
                next=cur.next
                cur.next=pre
                pre=cur
                cur=next
        return pre
            '''
	mid.next = None
	while(B):
		nextA = A.next
		A.next = B
		nextB = B.next
		B.next = nextA
		A = nextA
		B = nextB
	return head

#Reverse a linked list from position m to n. Do it in-place and in one-pass
# 1->2->3->4->5->NULL, m = 2 and n = 4, return 1->4->3->2->5->NULL

def reverseBetween(self, A, B, C):
        if C == B:
            return A
        revs , revsprev , revend , revendnext = None , None , None , None
        i = 1
        cur = A
        while cur and i <= C:
            if i < B:
                revsprev = cur
            if i == B:
                revs = cur
            if i == C:
                revend = cur
                revendnext = cur.next
            cur = cur.next
            i += 1
        revend.next = None
        revend = reverseLL(revs)
        if revsprev:
            revsprev.next = revend
        else:
            A = revend
        revs.next = revendnext
        return A 
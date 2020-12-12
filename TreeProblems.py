import math
import sys
import string
from collections import Counter
from collections import defaultdict
from collections import deque
'''
Given an inorder traversal of a cartesian tree, construct the tree.
Cartesian tree : is a heap ordered binary tree, where the root is greater than all the elements in the subtree. 
 Note: You may assume that duplicates do not exist in the tree. 
Now the soln approach: 
Inorder traversal : (Left tree) root (Right tree)
Note that the root is the max element in the whole array. Based on the info, can you figure out the position of the root in
 inorder traversal ? If so, can you separate out the elements which go in the left subtree and right subtree ?
Once you have the inorder traversal for left subtree, you can recursively solve for left subtree. Same for right subtree

We assume that a tree class is defined already and Treenode(value) will create a node with that val
'''
def Treebuild(A):
	if not A:
		return None
	root = max(A)
	ind = A.index(root)
	node = Treenode(root)
	node.left = Treebuild(A[:ind])
	node.right = Treebuild(A[ind+1:])
	return node

'''
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
Balanced tree : a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never
 differ by more than 1. 
 '''
 def arr_to_tree(A):
 	if not A:
 		return None
 	n = len(A)
 	mid = n/2
 	node = Treenode(A[mid])
 	node.left = arr_to_tree(A[:mid])
 	node.right = arr_to_tree(A[mid+1:])
 	return node

'''
Given inorder and postorder traversal of a tree, construct the binary tree.
Note: You may assume that duplicates do not exist in the tree. 
Example :
Input : 
        Inorder : [2, 1, 3]
        Postorder : [2, 3, 1]

Return : 
            1
           / \
          2   3
'''
def construct(A,B):# B is the postorder and A is the inorder
	if not B:
		return None
		'''
		Focus on the postorder traversal to begin with.
The last element in the traversal will definitely be the root.
Based on this information, can you identify the elements in the left subtree and right subtree
( Hint : Focus on inorder traversal and root information )

Once you do that, your problem has now been reduced to a smaller set. Now you have the inorder and postorder traversal
 for the left and right subtree and you need to figure them out.'''
	rootpos = A.index(B[-1])
	node = Treenode(B[-1])
	node.left = construct(A[:rootpos], B[:rootpos])
	node.right = construct(A[rootpos+1:], B[rootpos:-1])
	return node

# if A is inorder and B is preorder then B = [1,2,3], the code will look like:
def buildTree(self, A, B):
	    if not B:
	        return None
	    root_pos = B.index(A[0])
	    new_node = TreeNode(A[0])
	    new_node.left = self.buildTree(A[1:root_pos+1], B[:root_pos])
	    new_node.right = self.buildTree(A[root_pos+1:], B[root_pos+1:])
	    return new_node
'''
Given a binary tree, invert the binary tree and return it.
Given binary tree

     1
   /   \
  2     3
 / \   / \
4   5 6   7
invert and return

     1
   /   \
  3     2
 / \   / \
7   6 5   4
'''
def invertTree(root):
	if not root:
		return None
	root.left, root.right = invertTree(root.right),invertTree(root.left)
	return root
'''
Given a binary tree A with N nodes.

You have to remove all the half nodes and return the final binary tree.

NOTE:

Half nodes are nodes which have only one child.
Leaves should not be touched as they have both children as NULL

Input 1:

           1
         /   \
        2     3
       / 
      4

Input 2:

            1
          /   \
         2     3
        / \     \
       4   5     6


Example Output
Output 1:

           1
         /    \
        4      3
Output 2:

            1
          /   \
         2     6
        / \

       4   5
'''
def solve(n):# n is the root of the tree
        if not n:
            return None
        
        n.left = solve(n.left)
        n.right = solve(n.right)
        
        if bool(n.left) == bool(n.right):# no children there or both children there
            return n
            
        return n.right if not n.left else n.left# if left children not there then return
        # right, else return the left children
        # this is the step which is actually removing the half nodes recursively

'''
Given a Binary Tree A containing N nodes. You need to find the path from Root to a given node B.

No two nodes in the tree have same data values.
You can assume that B is present in the tree A and a path always exists.
Input 1:

 A =

           1
         /   \
        2     3
       / \   / \
      4   5 6   7 

B = 5

Input 2:

 A = 
            1
          /   \
         2     3
        / \ .   \
       4   5 .   6

B = 1

Example Output
Output 1:

 [1, 2, 5]
Output 2:

 [1]
 '''
def findpath(root,target,path):
	if not root:
		return None
	if root == target:
		return path+[root.val]
	left = findpath(root.left,target,path+[root.val])
	if left:
		return left
	right = findpath(root.right,target,path+[root.val])
	if right:
		return right
	else:
		return None
def solve(root,target):
	return findpath(root,target,[])

'''
Given a binary tree, determine if it is height-balanced.

 Height-balanced binary tree : is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1. 
Return 0 / 1 ( 0 for false, 1 for true ) for this problem'''

def ifbalanced(A):
	flag = 1
	def depth(root):
		if not root:
			return 0
		l = depth(root.left)
		r = depth(root.right)
		if abs(l-r) > 1:
			flag = 0
		return max(l,r) + 1
	depth(A)
	return flag
'''
Given two Binary Trees A and B, you need to merge them in a single binary tree.
The merge rule is that if two nodes overlap, then sum of node values is the new value of the merged node.
Otherwise, the non-null node will be used as the node of new tree'''

def merge(root1,root2):
	if not root1:
		return root2
	if not root2:
		return root1
	root1.val += root2.val
	root1.left = merge(root1.left,root2.left)
	root1.right = merge(root1.right,root2.right)
	return root1
'''
Given two binary trees, write a function to check if they are equal or not.
Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
Return 0 / 1 ( 0 for false, 1 for true ) for this problem'''
def isSameTree(A, B):
        if not A and not B:
            return True
        if not A or not B:
            return False
        
        return A.val == B.val and isSameTree(A.left,B.left) and isSameTree(A.right,B.right)
'''
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center)
Return 0 / 1 ( 0 for false, 1 for true ) for this problem.'''
def isSymmetric(A):
	def isexact(A,B):# this func checks if two trees are symmetric or not
		if A==None or B==None:
			return True
		if A==None or B==None:
			return False
		return A.val == B.val and isexact(A.left,B.right) and isexact(A.right,B.left)
	if not A:
		return 1
	if isexact(A.left,A.right):# we apply that for the subtrees of A
		return 1
	return 0
'''
Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all
# the values along the path equals the given sum'''
def sumpath(root,sum):
	if not root:
		return 0
	if root.val == sum and not root.left and not root.right:
		return 1
	if sumpath(root.left,sum-root.val):
		return 1
	if sumpath(root.right,sum-root.val):
		return 1
	return 0

#Given a binary tree and a sum, find all root-to-leaf paths where each path’s sum equals the given sum and return list od lists
def allpaths(root,target):
	paths = []
	def helper(root,target,curr_path):
		if root.left is None and root.right is None:
			if root.val = target:
				paths.append(curr_path+[root.val])
			return
		if root.left is not None:
			helper(root.left,target-root.val,curr_path+[root.val])
		if root.right is not None:
			helper(root.right,target-root.val,curr_path+ [root.val])
	if root is not None:
		helper(root,target,[])
	return paths

#Given a binary tree, find its minimum depth.
#The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
def minlenpath(A):
	if not A.left and not A.right:
		return 1
	return 1 + min(minlenpath(A.left) if A.left else 9e9, minlenpath(A.right) if A.right else 9e9)
	# if the right or left subtree does not exist then we attach a huge value 9e9 to that so that it gets ignored
	# if neither subtree is there then return 1 which is the dpeth of the tree in that case

#Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
#An example is the root-to-leaf path 1->2->3 which represents the number 123. Find the total sum of all root-to-leaf numbers % 1003.
def sumallpaths(A):
	sum = 0
	def helper(root,curr_path):
		if root.left is None and root.right is None:
			return sum += int(curr_path + str(root.val))
		if root.left:
			helper(root.left,curr_path+ str(root.val))
		if root.right:
			helper(root.right,curr_path + str(root.val))
	helper(A,"")
	return sum % 1003

#Given a binary tree, return the inorder traversal of its nodes’ values i.e. root.left -> root -> root.right
def printInorder(A):
	result = []
	stack = []
	node = A
	while(node or stack):
		if node:# this goes on until the leftmost node is reached after which node.left becomes none
			stack.append(node)
			node = node.left
		else:# when no more leftmost node left, start printing and move towards the root
			node = stack.pop()
			result.append(node.val)
			node = node.right# this is what is to be done for printing inorder, after the root, move right
			# stack will be empty after reaching the root at each level and then node = right node of the root
			# now again node is not none and we go to the if node part till we reach the leftmost node in that level
	return result

#Given a binary tree, return the postorder traversal of its nodes’ values, i.e. root.left-> root.right-> root
def printPostorder(A):
	result = []
	stack = [A]
	while(stack):
		node = stack.pop()
		result.append(node.val)# this will create the res in the reverse order, so need to return the reverse of the ans
		if node.left:# if  there is a left subtree, put it in stack
			stack.append(node.left)
		if node.right:# the right subtree will be processed first as it will be on top of stack, thus creating the reverse order
			stack.append(node.right)# after this goes to the while loop to check if stack is empty and acts suitably
	return result[::-1]

#Given a binary tree, return the preorder traversal of its nodes’ values, i.e. root -> root.left-> root.right
def printPreorder(A):
	result = []
	stack = [A]
	while(stack):
		node = stack.pop()
		result.append(node.val)
		if node.right:# if  there is a right subtree, put it in stack
			stack.append(node.right)
		if node.left:# the left subtree will be processed first as it will be on top of stack, thus creating the right order
			stack.append(node.left)# after this goes to the while loop to check if stack is empty and acts suitably
	return result

#Given a binary tree A consisting of N nodes, return a 2-D array denoting the vertical order traversal of A.
'''
   6
    /   \
   3     7
  / \     \
 2   5     9
 output: [
    [2],
    [3],
    [6, 5],
    [7],
    [9]
 ]
 Nodes on Vertical Line 1: 2
 Nodes on Vertical Line 2: 3
 Nodes on Vertical Line 3: 6, 5
 As 6 and 5 are on the same vertical level but as 6 comes first in the pre-order traversal of the tree so we will output 6 before 5.
 Nodes on Vertical Line 4: 7
 Nodes on Vertical Line 5: 9
 SOlution approach:An efficient solution based on hash map is discussed. We need to check the Horizontal Distances from root for all nodes.
  If two nodes have the same Horizontal Distance (HD), then they are on same vertical line. The idea of HD is simple. HD for root is 0,
   a right edge (edge connecting to right subtree) is considered as +1 horizontal distance and a left edge is considered as -1 horizontal 
   distance. For example, in the above tree, HD for Node 2 is at -2, HD for Node 3 is -1, HD for 5 is 0, HD for node 7 is +1 and for
    node 9 is +2.
We can do level order traversal of the given Binary Tree. While traversing the tree, we can maintain HDs. We initially pass the
 horizontal distance as 0 for root. For left subtree, we pass the Horizontal Distance as Horizontal distance of root minus 1.
  For right subtree, we pass the Horizontal Distance as Horizontal Distance of root plus 1. For every HD value, we maintain a
  list of nodes in a hasp map. Whenever we see a node in traversal, we go to the hash map entry and add the node to the hash map 
  using HD as a key in map.'''
 def verticalTraversal(A):
 	cols = defaultdict(list)# dict so that at each key we have a list
 	queue = [(A,0)]
 	for node,i in queue:# here node = None is possible, it will be checked in next line
 		if node:# this will be empty when all the cols are filled with the reqd vals
 			col[i].append(node.val)# append the value to the ith list which looks like the ith col when seen vertically
 			queue += (node.left,i-1),(node.right,i+1)# for each node we update the queue with its children
 	return [cols[i] for i in sorted(cols)]# dict returned as a list of lists

#Given a binary search tree, write a function to find the kth smallest element in the tree
'''Soln approach:
Note the property of the binary search tree.
All elements smaller than root will be in the left subtree, and all elements greater than root will be in the right subtree.
This means we need not even explore the right subtree till we have explored everything in the left subtree. Or in other words,
 we go to the right subtree only when the size of left subtree + 1 ( root ) < k.

With that in mind, we can come up with an easy recursive solution which is similar to inorder traversal :

Step 1: Find the kth smallest element in left subtree decrementing k for every node visited. If answer is found, return answer.
Step 2: Decrement k by 1. If k == 0 ( this node is the kth node visited ), return node’s value
Step 3: Find the kth smallest element in right subtree.'''
def kthsmallest(A,k):
	stack = []
	node = A
	while(stack or node):
		if node:
			stack.append(node)
			node = node.left
		else:
			node = stack.pop()
			k -= 1
			if k == 0:
				return node.val
			node = node.right
	return None
'''
Given a binary search tree T, where each node contains a positive integer, and an integer K, you have to find whether or not there
 exist two different nodes A and B such that A.value + B.value = K
 soln approach:
 If you do inorder traversal of BST you visit elements in increasing order. So, we use a two pointer approach, where we keep two pointers
  pt1 and pt2. Initially pt1 is at smallest value and pt2 at largest value.
Then we compare sum = pt1.value + pt2.value. If sum < target, we increase pt2, else we decrease pt2 until we reach target.'''
def t2sum(A,target):
	L = []
	def inorder(root):
		inorder(root.left)
		L.append(root.val)
		inorder(root.right)
	inorder(A)
	i,j = 0,len(L)-1
	while(i<j and j>=0 and i< len(L)):
		find = target-L[i]
		if L[j] > find:
			j -= 1
		if L[j] == find:
			return 1
		if L[j] < find:
			i += 1
	return 0
'''
Find shortest unique prefix to represent each word in the list.

Example:

Input: [zebra, dog, duck, dove]
Output: {z, dog, du, dov}
where we can see that
zebra = z
dog = dog
duck = du
dove = dov
 Assume that no word is prefix of another. In other words, the representation is always possible. 

input: ["zebra", "dog", "duck", "dot"]

Now we will build prefix tree and we will also store count of characters

                root
                /|
         (d, 3)/ |(z, 1)
              /  |
          Node1  Node2
           /|        \
     (o,2)/ |(u,1)    \(e,1)
         /  |          \
   Node1.1  Node1.2     Node2.1
      | \         \            \
(g,1) |  \ (t,1)   \(c,1)       \(b,1)
      |   \         \            \ 
    Leaf Leaf       Node1.2.1     Node2.1.1
    (dog)  (dot)        \                  \
                         \(k, 1)            \(r, 1)
                          \                  \   
                          Leaf               Node2.1.1.1
                          (duck)                       \
                                                        \(a,1)
                                                         \
                                                         Leaf
                                                         (zebra)

Now, for every leaf / word , we find the character nearest to the root with frequency as 1. 
The prefix that the path from root to this character corresponds to, is the representation of the word. '''
def prefix(words):
	prefixes = defaultdict(int)# creating empty dict
	for word in words:# now making the dict ready i.e. creating the reqd trie
		for i in range(len(word)):
			prefixes[word[:i]] += 1
	result = []# trie created and now initialize the ans
	for word in words:
		temp_ans = word# this will be returned if now prefix found
		for i in range(len(word)):
			#pref = word[:i]
			if prefixes[word[:i]] == 1:# character nearest to the root with frequency as 1 found
				temp_ans = word[:i]# this is the reqd prefix
				break
		result.append(temp_ans)# temp ans is the ans for this word
	return result# final list containing all the temp ans from all the words

'''
You are given the following :

A positive number N
Heights : A list of heights of N persons standing in a queue
Infronts : A list of numbers corresponding to each person (P) that gives the number of persons who are taller than P and standing
 in front of P
You need to return list of actual order of persons’s height

Consider that heights will be unique

Example

Input : 
Heights: 5 3 2 6 1 4
InFronts: 0 1 2 0 3 2
Output : 
actual order is: 5 3 2 1 6 4

The Solution Approach: Iterate from shortest to tallest after sorting the heights. In each iteration, place the curr person in the correct 
position. Observe that the curr person is taller than all prev people since all heights distinct. So at evert iter, we find
a place such that num of empty pso in front = infront val of this person. If we cannot find such a pos then this is not possible.'''
def order(height,infront):
	locations = [i for i in range(len(height))]
	answer = [0]*len(height)# this is the answer list in the right order initialized to 0 in the beginning for all pos
	zippedlist = list(sorted(zip(height,infront)))# this is the list we wanted and we have to iter over this, on every iteration
	for i in zippedlist:# i[0] is the elem which have to placed in the correct place, next elem always taller than the curr, since sorted
	# the 2nd coordinate which is i[1] will determine where to put this i[0]
		pos = locations.pop(i[1])# finding the correct pos where to put, so that the number of empty pos in front is i[1]
		answer[pos] = i[0]# putting the i[0] in the correct place
	return answer

'''
Find the lowest common ancestor in an unordered binary tree given two values in the tree.

 Lowest common ancestor : the lowest common ancestor (LCA) of two nodes v and w in a tree or directed acyclic graph (DAG) is the
  lowest (i.e. deepest) node that has both v and w as descendants. 
Example :


        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2_     0        8
         /   \
         7    4
For the above tree, the LCA of nodes 5 and 1 is 3.

 LCA = Lowest common ancestor 
Please note that LCA for nodes 5 and 4 is 5.

You are given 2 values. Find the lowest common ancestor of the two nodes represented by val1 and val2
No guarantee that val1 and val2 exist in the tree. If one value doesn’t exist in the tree then return -1.
There are no duplicate values.
You can use extra memory, helper functions, and can modify the node struct but, you can’t add a parent pointer.
Solution Approach:
Linear solution using path calculation :

1) Find path from root to n1 and store it in a vector or array.
2) Find path from root to n2 and store it in another vector or array.
3) Traverse both paths till the values in arrays are same. Return the common element just before the mismatch'''
def getpath(root,val): # this does 1) and 2)
	if not root:
		return []
	if root.val == val:
		return [root]
	left = getpath(root.left,val)
	right = getpath(root.right,val)
	if left:
		return [root] + left
	if right:
		return [root] + right
	return []
def LCA(root,val1,val2):# this does 3)
	path1 = getpath(root,val1)
	path2 = getpath(root,val2)
	lca = -1
	for a,b in zip(path1,path2):# zip will put together two lists as one list whose len is same as the shorter one of path1 and path2
		if a != b:# the first where it breaks is the branch before which lca lies, we have stored it in lca
			break
		lca = a.val# update after the check, so that we recover the point after which the equality breaks
	return lca
'''
Given a Binary Tree A consisting of N nodes.

You need to find all the cousins of node B.

NOTE:

Siblings should not be considered as cousins.
Try to do it in single traversal.
You can assume that Node B is there in the tree A.
Order doesn't matter in the output.'''
def printcousins(root,node_to_find):
	if not root:
		return []
	ans = []
	if root.val == node_to_find:
		return ans
		'''The idea is to go for level order traversal of the tree, as the cousins and siblings of a node can be found in its level order
		 traversal. Run the traversal till the level containing the node is not found, and if found, print the given level.


How to print the cousin nodes instead of siblings and how to get the nodes of that level in the queue?

During the level order, when for the parent node, if parent->left == Node_to_find, or parent->right == Node_to_find, then the children
 of this parent must not be pushed into the queue (as one is the node and other will be its sibling). Push the remaining nodes at the
  same level in the queue and then exit the loop. The current queue will have the nodes at the next level (the level of the node being
   searched, except the node and its sibling). Now, print the queue.

This is a single level order traversal, hence time complexity = O(N), and Auxiliary space = O(N)'''
	queue = [root]
	found = 0
	while(queue and found == 0):
		size = len(queue)
		while(size):
			curr_node = queue[0]
			queue.pop(0)
			if ((curr_node.left and curr_node.left.val == node_to_find) or (curr_node.right and curr_node.right.val == node_to_find)):
				found = 1
			else:
				if curr_node.left:
					queue.append(curr_node.left)
				if curr_node.right:
					queue.append(curr_node.right)
			size -= 1
	for i in range(len(queue)):
		ans.append(queue[i].val)
	return ans
'''
Given a binary tree, return the zigzag level order traversal of its nodes’ values. (ie, from left to right, then right to left for the next level and alternate between).

Example :
Given binary tree

    3
   / \
  9  20
    /  \
   15   7
return

[
         [3],
         [20, 9],
         [15, 7]
]
Soln : We will be using 2 stacks to solve this problem. One for the current layer and other one for the next layer.
 Also keep a flag which indicates the direction of traversal on any level.

You need to pop out the elements from current layer stack and depending upon the value of flag push the childs of current element
 in next layer stack. You should maintain the output sequence in the process as well. Remember to swap the stacks before next iteration.'''
def zigzagLevelOrder(self, root):
        result=[]
        level=-1
        queue=deque()
        queue.append(root)
        while queue:
            temp=[]
            level+=1
            nodes=len(queue)
            while nodes>0:
                node=queue.popleft()
                temp.append(node.val)
                nodes-=1
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if level%2:
                result.append(temp[::-1])
            else:
                result.append(temp)
        return result
'''
Given a binary tree, flatten it to a linked list in-place.

Example :
Given


         1
        / \
       2   5
      / \   \
     3   4   6
The flattened tree should look like:

   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
Note that the left child of all nodes should be NULL.
Solution Approach: If you notice carefully in the flattened tree, each node’s right child points to the next node of a pre-order traversal.

Now, if a node does not have left node, then our problem reduces to solving it for the node->right.
If it does, then the next element in the preorder traversal is the immediate left child. But if we make the immediate left child as
 the right child of the node, then the right subtree will be lost. So we figure out where the right subtree should go. In the preorder
  traversal, the right subtree comes right after the rightmost element in the left subtree.
So we figure out the rightmost element in the left subtree, and attach the right subtree as its right child. We make the left child
 as the right child now and move on to the next node'''
def flatten(self, A):
        current = A
        stack = deque()
        
        while(current or stack):
            
            if(current.right):
                stack.append(current.right)
            if(current.left == None and stack):
                current.right = stack.pop()
            else:
                current.right = current.left
            current.left = None
            current = current.right
        
        return A

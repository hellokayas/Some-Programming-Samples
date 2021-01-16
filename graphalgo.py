import math
import sys
import string
import collections
from collections import Counter
from collections import defaultdict
from collections import deque
from bisect import bisect_left
from collections import defaultdict
import heapq
'''
There is a rectangle with left bottom as  (0, 0) and right up as (x, y). There are N circles such that their centers are inside the rectangle.
Radius of each circle is R. Now we need to find out if it is possible that we can move from (0, 0) to (x, y) without touching any circle.

Note : We can move from any cell to any of its 8 adjecent neighbours and we cannot move outside the boundary of the rectangle at any
 point of time.
soln: Check if (i,j) is a valid point for all 0<=i<=x, 0<=j<=y. By valid point we mean that none of the circle should contain it.

Now you know all the valid point in rectangle. You need to figure out if you can go from (0,0) to (x,y) through valid points.
 This can be done with any graph traversal algorithms like BFS/DFS. 
 1st argument given is an Integer x.
2nd argument given is an Integer y.
3rd argument given is an Integer N, number of circles.
4th argument given is an Integer R, radius of each circle.
5th argument given is an Array A of size N, where A[i] = x cordinate of ith circle
6th argument given is an Array B of size N, where B[i] = y cordinate of ith circle'''
def solve(X,Y,N,R,A,B):
	centers = [x for x in zip(A,B)]
	moves = {(-1,0),(1,0),(0,1),(0,-1),(-1,1),(1,1),(-1,-1),(1,-1)}
	def outside_circles(x,y):
		for center in centers:
			if (x-center[0])**2 + (y-center[1])**2 <= R**2:
				return False
		return True
	def is_safe(x,y):
		return x >=0 and x <= X and y >= 0 and y <= Y and outside_circles(x,y)
	visited = [[0 for _ in range(Y+1)] for _ in range(X+1)]
	visited[0][0] = 1
	queue = [(0,0)]
	while queue:
		x,y = queue.pop(0)
		if x == X and y == Y:
			return "Yes"
		for i in moves:
			nextx,nexty = x + i[0],y + i[1]
			if is_safe(nextx,nexty) and not visited[nextx][nexty]:
				visited[nextx][nexty] = 1
				queue.append((nextx,nexty))
	return "No"
'''
Given a binary matrix A of size N x M.
Cells which contain 1 are called filled cell and cell that contain 0 are called empty cell.
Two cells are said to be connected if they are adjacent to each other horizontally, vertically, or diagonally.
If one or more filled cells are also connected, they form a region. Find the length of the largest region
Soln app: Just perform dfs from each unvisited one and count the number of ones which can be visited from here and maintain the maximum.
A cell in 2D matrix can be connected to at most 8 neighbors. So in DFS, we make recursive calls for 8 neighbors. We keep track of the
 visited 1’s in every DFS and update maximum length region. Time complexity: O(N x M)'''
def region(A):
	res = 0# initialize the final result
	m = len(A)# the num of rows
	n = len(A[0])# the num of cols
	for i in range(m):
		for j in range(n):
			if not A[i][j]:# if the cell is 0, just skip and go to the next cell
				continue
			temp = 0# we have found a 1 and now we will do a dfs from here to find the local max and store that in temp
			queue = [(i,j)]# initialize the q and (i,j) cell contains the 1
			A[i][j] = 0# make it 0 permanently so that we do not encounter anymore again and the algorithm terminates
			while queue:# the dfs begins
				r,c = queue.pop()
				temp += 1# increasing temp as we encounter 1's 
				for dr, dc in (-1, -1), (-1, 0), (-1, 1), (0, -1),(0, 1), (1, -1), (1, 0), (1, 1):
					if 0 <= r+dr < m and 0 <= c+dc < n and A[r+dr][c+dc]:# all the 8 possible directions checking for 1
						A[r+dr][c+dc] = 0# change it to 0 so that we do not overcount
						queue.append((r+dr, c+dc))# standard step in dfs
			res = max(res,temp)# updating the global max using each local max
	return res
'''
Given a binary tree, return the level order traversal of its nodes’ values. (ie, from left to right, level by level)
Example :
Given binary tree

    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:

[
  [3],
  [9,20],
  [15,7]
]
Also think about a version of the question where you are asked to do a level order traversal of the tree when depth of the tree
 is much greater than number of nodes on a level.'''
def levelTraversal(A):
	res = []# initialize the 2d list
	queue = deque([A])# append the starting node = root
	while queue:
		res.append([])# at the beginning of each level append an empty list in which we will put all the nodes at that level
		len_level = len(queue)
		for _ in range(len_level):# for each level
			node = queue.popleft()# nodes kept in the queue from the prev level
			res[-1].append(node.val)# append the val and process the other nodes in ths level similarly
			if node.left:
				queue.append(node.left)# append the left and right children for processing int the next level
			if node.right:
				queue.append(node.right)
	return res
'''
RULES:

The game is played with cubic dice of 6 faces numbered from 1 to A.
Starting from 1 , land on square 100 with the exact roll of the die. If moving the number rolled would place the player beyond square 100, no move is made.
If a player lands at the base of a ladder, the player must climb the ladder. Ladders go up only.
If a player lands at the mouth of a snake, the player must go down the snake and come out through the tail. Snakes go down only.
BOARD DESCRIPTION:

The board is always 10 x 10 with squares numbered from 1 to 100.
The board contains N ladders given in a form of 2D matrix A of size N * 2 where (A[i][0], A[i][1]) denotes a ladder that has its base on square A[i][0] and end at square A[i][1].
The board contains M snakes given in a form of 2D matrix B of size M * 2 where (B[i][0], B[i][1]) denotes a snake that has its mouth on square B[i][0] and tail at square B[i][1].
Soln approach: Let’s model the board as a graph. Every square on the board is a node. The source node is square 1. The destination node is square 100. From every square, it is possible to reach any of the 6 squares in front of it in one move. So every square has a directed edge to the 6 squares in front of it.

For a snake starting at square i and finishing at j, we can consider that there is no node with index i in the graph. Because
reaching node i is equivalent to reaching node j since the snake at node i will immediately take us down to node j. The same
analogy goes for ladders too. To handle the snakes and ladders, let’s keep an array go_immediately_to[] Now let’s run a standard Breadth First Search(BFS). Whenever we reach a node i, we will consider that we have reached the node go_immediately_to[] and then continue with the BFS as usual. The distance of the destination is the solution to the problem.

Time Complexity:
The size of the board is always 10 x 10. You can consider time complexity of each BFS is constant as the number of snakes or ladders
 won’t have much effect.'''
def snakes_ladders(A,B):
	snakes = {}
	ladders = {}
	for i,j in A:
		snakes[i] = j
	for i,j in B:
		ladders[i] = j
	queue = deque([1])
	steps = 0
	seen = set()
	while queue:
		for _ in range(len(queue)):
			curr = queue.popleft()
			if curr == 100:
				return steps
			for i in range(1,7):
				if curr+i >100 or curr + i in seen:
					continue
				if curr+i in ladders:
					nex = ladders[curr+i]
				if curr+i in snakes:
					nex = snakes[curr+i]
				else:
					nex = curr + i
				seen.add(nex)
				queue.append(nex)
		steps += 1
	return -1
'''
Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
 A height balanced BST : a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node 
 never differ by more than 1.
soln : convert the LL to a list and then convert that list to ht balanced BST'''
def LL_to_list(A):
	L = []
	while A:
		L.append(A.val)
		A = A.next
	return L
def list_to_BST(L):
	if not L:
		return None
	n = len(L)
	mid = n//2
	root = Node(L[mid])
	root.left = list_to_BST(L[:mid])
	root.right = list_to_BST(L[mid+1:])
	return root
'''
Given any source point, (C, D) and destination point, (E, F) on a chess board, we need to find whether Knight can move to the destination
 or not. The first argument of input contains an integer A.The second argument of input contains an integer B.
    => The chessboard is of size A x B.
The third argument of input contains an integer C. The fourth argument of input contains an integer D.
    => The Knight is initially at position (C, D).
The fifth argument of input contains an integer E. The sixth argument of input contains an integer F.
    => The Knight wants to reach position (E, F).
A knight can move to 8 positions from (x,y). 

(x, y) -> 
    (x + 2, y + 1)  
    (x + 2, y - 1)
    (x - 2, y + 1)
    (x - 2, y - 1)
    (x + 1, y + 2)
    (x + 1, y - 2)
    (x - 1, y + 2)
    (x - 1, y - 2)

Corresponding to the knight's move, we can define edges. 
(x,y) will have an edge to the 8 neighbors defined above. 

To find the shortest path, we just run a plain BFS. '''
def knight(A, B, C, D, E, F):
        from collections import deque
        seen = set()
        q = deque([(C, D)])
        res = 0
        while q:
            for _ in range(len(q)):
                i, j = q.popleft()
                if i == E and j == F:
                    return res
                for di, dj in (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2):
                    if 1 <= i + di <= A and 1 <= j + dj <= B and (i+di, j+dj) not in seen:
                        seen.add((i+di, j+dj))
                        q.append((i+di, j+dj))
            res += 1
        return -1
'''
There are A islands and there are M bridges connecting them. Each bridge has some cost attached to it.
We need to find bridges with minimal cost such that all islands are connected.
It is guaranteed that input data will contain at least one possible scenario in which all islands are connected with each other
Soln: Prim's algo for MST'''
def solve(A, B):
        #prim
        connected = {1}# this will grow up to form the MST
        adj = defaultdict(list)
    
        for u, v, w in B:
            adj[u].append( (w,  v))
            adj[v].append( ( w, u))
        pending = adj[1][:]
        heapq.heapify(pending)
        cost = 0
        while len(pending) > 0:
            w, u = heapq.heappop(pending)
            if u in connected:
                continue
            connected.add(u)
            cost += w
            for ww, v in adj[u]:
                if v not in connected:
                    heapq.heappush(pending,  (ww, v) )
        return cost
'''
There are a total of A courses you have to take, labeled from 1 to A.
Some courses may have prerequisites, for example to take course 2 you have to first take course 1, which is expressed as a pair: [1,2].
Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
Return 1 if it is possible to finish all the courses, or 0 if it is not possible to finish all the courses

Soln : Consider a graph with courses from 1 to N representing the nodes of the graph and each prerequisite pair [u, v] correspond to a directed edge from u to v.
It is obvious that we will get several disjoint components of the graph.
The problem reduces down to finding a directed cycle in the whole graph. If any such cycle is present, it is not possible to finish all the courses.
Depth First Traversal(DFS) can be used to detect cycle in a Graph. There is a cycle in a graph only if there is a back edge present in the graph. A back edge is an edge that is from a node to itself (self loop) or one 
of its ancestor in the tree produced by DFS.
For a disconnected graph, we can check for cycle in individual DFS trees by checking back edges'''
def solve(A, B, C):
        
        visited = [0 for _ in range(A+1)]
        stack = [0 for _ in range(A+1)]
        edges = {}
        for i in range(len(B)):
            if B[i] not in edges:
                edges[B[i]] = [C[i]]
            else:
                edges[B[i]].append(C[i])
        
        def isCycle(node):
            visited[node] = 1
            stack[node] = 1
            
            if node in edges:
                for x in edges[node]:
                    if visited[x] == 0:
                        if isCycle(x):
                            return True
                    elif stack[x] == 1:# this is the back-edge
                        return True
                
            stack[node] = 0
            return False
        
        for i in range(1 , A+1):
            if visited[i] == 0:
                if isCycle(i):
                    return 0
        
        return 1
'''
Given an undirected graph having A nodes labelled from 1 to A with M edges given in a form of matrix B of size M x 2 where 
(B[i][0], B[i][1]) represents two nodes B[i][0] and B[i][1] connected by an edge.
Find whether the graph contains a cycle or not, return 1 if cycle is present else return 0
Soln : Like directed graphs, we can use DFS to detect cycle in an undirected graph in O(A+M) time.
We do a DFS traversal of the given graph. For every visited vertex ‘v’, if there is an adjacent ‘u’ such that u is already visited and u is not parent of v, then there is a cycle in graph.
If we don’t find such an adjacent for any vertex, we say that there is no cycle.
The assumption of this approach is that there are no parallel edges between any two vertices'''
def dfs(node,parent,visited):
	visited[node] = True
	for i in graph[node]:# so i is an adjacent node of the curr node
		if visited[i] == True and i != parent:# this means backedge
			return True
		elif visited[i] == False and dfs(i,node,visited) == True:# no problem with curr node but found cycle later in the recursive calls
			return True
	return False# found no cycles
def detectCycle(vertices,edges):
	graph = defaultdict(list)
	visited = {}
	for i in edges:# the next four lines initialize the graph and the visited dict which stores whether a vertex is seen or not
		graph[i[0]].append(i[1])
		graph[i[1]].append(i[0])
		visited[i[0]] = False
		visited[i[1]] = False
	for i in visited.keys():# now iterate over all the vertices(cannot pass vertices in place of visited.keys() sue to type mismatch)
		if visited[i] == False:# so that we dont go over the same vertices and overcount
			if dfs(i,-1,visited) == True:# parent of a node where dfs is starting is initialized to -1
				return 1# cycle detected
	return 0
'''
Given N x M character matrix A of O's and X's, where O = white, X = black.
Return the number of black shapes. A black shape consists of one or more adjacent X's (diagonals not included)
Answer := 0
Loop i = 1 to N :
    Loop j = 1 to M:
          IF MATRIX at i, j equal to 'X' and not visited:
                 BFS/DFS to mark the connected area as visited
                 update Answer
    EndLoop
EndLoop

return Answer'''
def black(A):
        n = len(A)
        m = len(A[0])
        visited = [[False for i in range(m)] for h in range(n)]

        def is_valid(r, c):
            if 0<=r<n and 0<=c<m:
                return True
            return False
        res = 0

        def dfs(i, j):
            if A[i][j] == "X" and visited[i][j] == False:
                    visited[i][j] = True
                    for k in [(1,0), (0, 1), (-1,0), (0, -1)]:
                        if is_valid(i+k[0], j+k[1]):
                            dfs(i+k[0], j+k[1])
                    return 1
            return 0

        for x in range(n):
            for y in range(m):    
                if dfs(x, y)==1:
                    res += 1
        return res
# Determine if a graph is Bipartite Graph using DFS
# Class to represent a graph object
class Graph:
    # Constructor
    def __init__(self, edges, N):
 
        # A List of Lists to represent an adjacency list
        self.adjList = [[] for _ in range(N)]
 
        # add edges to the undirected graph
        for (src, dest) in edges:
 
            self.adjList[src].append(dest)
            self.adjList[dest].append(src)
 
 
# Perform DFS on graph starting from vertex v
def DFS(graph, v, discovered, color):# this is the actual function which you need to write 
 
    # do for every edge (v -> u)
    for u in graph.adjList[v]:
 
        # if vertex u is explored for first time
        if not discovered[u]:
 
            # mark current node as discovered
            discovered[u] = True
 
            # set color as opposite color of parent node
            color[u] = not color[v]
 
            # if DFS on any subtree rooted at v we return False
            if not DFS(graph, u, discovered, color):
                return False
 
        # if the vertex is already been discovered and color of
        # vertex u and v are same, then the graph is not Bipartite
        elif color[v] == color[u]:
            return False
 
    return True
 
 
if __name__ == '__main__':
 
    # List of graph edges as per above diagram
    edges = [
        (1, 2), (2, 3), (2, 8), (3, 4), (4, 6), (5, 7), (5, 9), (8, 9), (2, 4)
        # if we remove 2->4 edge, graph is becomes Bipartite
    ]
 
    # Set number of vertices in the graph
    N = 10
 
    # create a graph from edges
    graph = Graph(edges, N)
 
    # stores vertex is discovered or not
    discovered = [False] * N
 
    # stores color 0 or 1 of each vertex in DFS
    color = [False] * N
 
    # mark source vertex as discovered and
    # set its color to 0
    discovered[0] = True
    color[0] = False
 
    # start DFS traversal from any node as graph
    # is connected and undirected
    if DFS(graph, 1, discovered, color):
        print("Bipartite Graph")
    else:
        print("Not a Bipartite Graph")
'''
Given an arbitrary unweighted rooted tree which consists of N nodes.
The goal of the problem is to find largest distance between two nodes in a tree.
Distance between two nodes is a number of edges on a path between the nodes (there will be a unique path between any pair of nodes since it is a tree).
The nodes will be numbered 0 through N - 1.The tree is given as an array A, there is an edge between nodes A[i] and i (0 <= i < N). 
Exactly one of the i's will have A[i] equal to -1, it will be root node.

Soln: Let u be the arbitrary vertex. We have a schematic like

    u
    |
    |
    |
    x
   / \
  /   \
 /     \
s       t ,
where x is the junction of s, t, u (i.e. the unique vertex that lies on each of the three paths between these vertices).

Suppose that v is a vertex maximally distant from u. If the schematic now looks like

    u
    |
    |
    |
    x   v
   / \ /
  /   *
 /     \
s       t ,
then

d(s, t) = d(s, x) + d(x, t) <= d(s, x) + d(x, v) = d(s, v),
where the inequality holds because d(u, t) = d(u, x) + d(x, t) and d(u, v) = d(u, x) + d(x, v). There is a symmetric case where v attaches between s and x instead of between x and t.

The other case looks like

    u
    |
    *---v
    |
    x
   / \
  /   \
 /     \
s       t .
Now,

d(u, s) <= d(u, v) <= d(u, x) + d(x, v)
d(u, t) <= d(u, v) <= d(u, x) + d(x, v)

d(s, t)  = d(s, x) + d(x, t)
         = d(u, s) + d(u, t) - 2 d(u, x)
        <= 2 d(x, v)

2 d(s, t) <= d(s, t) + 2 d(x, v)
           = d(s, x) + d(x, v) + d(v, x) + d(x, t)
           = d(v, s) + d(v, t),
so max(d(v, s), d(v, t)) >= d(s, t) by an averaging argument, and v belongs to a maximally distant pair.

perform BFS to find a node which is farthest away from it and then perform BFS on that node.
 The greatest distance from the second BFS will yield the diameter.'''

    def LongestPathLength(self):
 
        # first BFS to find one end point of longest path
        node, Dis = self.BFS(0)
 
        # second BFS to find the actual longest path
        node_2, LongDis = self.BFS(node)
 
        print('Longest path is from', node, 'to', node_2, 'of length', LongDis)

# Intialisation of graph inside some class Graph
    def __init__(self, vertices):
 
        # No. of vertices
        self.vertices = vertices
 
        # adjacency list
        self.adj = {i: [] for i in range(self.vertices)}
 
    def addEdge(self, u, v):
        # add u to v's list
        self.adj[u].append(v)
        # since the graph is undirected
        self.adj[v].append(u)

# method return farthest node and its distance from node u, this is the actual function doing everything
    def BFS(u):
        # marking all nodes as unvisited
        visited = [False for i in range(vertices + 1)]
        # mark all distance with -1
        distance = [-1 for i in range(vertices + 1)]
 
        # distance of u from u will be 0
        distance[u] = 0
        # in-built library for queue which performs fast oprations on both the ends
        queue = deque()
        queue.append(u)
        # mark node u as visited
        visited[u] = True
 
        while queue:
 
            # pop the front of the queue(0th element)
            front = queue.popleft()
            # loop for all adjacent nodes of node front
 
            for i in adj[front]:
                if not visited[i]:
                    # mark the ith node as visited
                    visited[i] = True
                    # make distance of i , one more than distance of front
                    distance[i] = distance[front]+1
                    # Push node into the stack only if it is not visited already
                    queue.append(i)
 
        maxDis = 0
 
        # get farthest node distance and its index
        for i in range(vertices):
            if distance[i] > maxDis:
 
                maxDis = distance[i]
                nodeIdx = i
 
        return nodeIdx, maxDis
'''
Given a 2D character matrix A of size N x M, containing 'X' and 'O', capture all regions surrounded by 'X'.
A region is captured by flipping all 'O's into 'X's in that surrounded region

Soln: We already know chunks of O which remain as O are the ones which have at least one O connected to them which is on the boundary.
Use BFS starting from ‘O’s on the boundary and mark them as ‘B’, then iterate over the whole board and mark ‘O’ as ‘X’ and ‘B’ as ‘O’.'''
def ourbfs(A,i,j):
	if 0<= i < len(A) and 0<= j < len(A[0]):
		if A[i][j] == "O":
			A[i][j] = "B"
			ourbfs(A,i+1,j)
			ourbfs(A,i-1,j)
			ourbfs(A,i,j+1)
			ourbfs(A,i,j-1)
def captureRegion(A):
	for j in range(len(A[0])):
		if A[0][j] == "O":
			ourbfs(A,0,j)
		if A[-1][j] == "O":
			ourbfs(A,len(A)-1,j)
	for i in range(len(A)):
		if A[i][0] == "O":
			ourbfs(A,i,0)
		if A[i][-1] == "O":
			ourbfs(A,i,len(A[0])-1)
	for i in range(A):
		for j in range(len(A[0])):
			if A[i][j] == "O":
				A[i][j] = "X"
			if A[i][j] == "B":
				A[i][j] = "O"
'''
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The cell itself does not count as an adjacent cell.
The same letter cell may be used more than once.

Example :

Given board =

[
  ["ABCE"],
  ["SFCS"],
  ["ADEE"]
]
word = "ABCCED", -> returns 1,
word = "SEE", -> returns 1,
word = "ABCB", -> returns 1,
word = "ABFSAB" -> returns 1
word = "ABCD" -> returns 0
Note that 1 corresponds to true, and 0 corresponds to false.

Soln: You iterate over every cell of the matrix to explore if it could be the starting point. Then for every neighboring character which has the same character as the next character in the string, we explore if rest of the string can be formed using that neighbor cell as the starting point.

To sum it up,
exist(board, word, i , j) is true if for any neighbor (k,l) of (i,j)
exist(board, word[1:], k, l) is true

Now note that we could memoize the answer for exist(board, word suffix length, i, j).'''
def exist(A, B):
        m, n = len(A), len(A[0])
        
        def helper(i, j, k):
            if 0 <= i < m and 0 <= j < n and A[i][j] == B[k]:
                if k == len(B)-1:
                    return True
                for di, dj in (-1, 0), (1, 0), (0, -1), (0, 1):
                    if helper(i+di, j+dj, k+1):
                        return True
            
        for i in range(m):
            for j in range(n):
                if helper(i, j , 0):
                    return 1
        return 0
'''
Given an directed graph having A nodes labelled from 1 to A containing M edges given by matrix B of size M x 2such that there is a edge directed from node

B[i][0] to node B[i][1].

Find whether a path exists from node 1 to node A.

Return 1 if path exists else return 0.

NOTE:

There are no self-loops in the graph.
There are no multiple edges between two nodes.
The graph may or may not be connected.
Nodes are numbered from 1 to A.
Your solution will run on multiple test cases. If you are using global variables make sure to clear them.'''
def solve(t, E):
        es = defaultdict(list)
        
        for e in E:
            es[e[0]].append(e[1])
        
        stack = [1]
        visited = set([1])
        while stack:
            u = stack.pop()
            for v in es[u]:
                if not v in visited:
                    stack.append(v)
                    visited.add(v)
                    
        return 1 if t in visited else 0
'''
Given two words A and B, and a dictionary, C, find the length of shortest transformation sequence from A to B, such that:

You must change exactly one character in every transformation.
Each intermediate word must exist in the dictionary.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters'''

# Returns length of shortest chain
# to reach 'target' from 'start'
# using minimum number of adjacent
# moves. D is dictionary
def shortestChainLen(start, target, D):
     
    # If the target is not
    # present in the dictionary
    if target not in D:
        return 0
 
    # To store the current chain length
    # and the length of the words
    level, wordlength = 0, len(start)
 
    # Push the starting word into the queue
    Q =  deque()
    Q.append(start)
 
    # While the queue is non-empty
    while (len(Q) > 0):
         
        # Increment the chain length
        level += 1
 
        # Current size of the queue
        sizeofQ = len(Q)
 
        # Since the queue is being updated while
        # it is being traversed so only the
        # elements which were already present
        # in the queue before the start of this
        # loop will be traversed for now
        for i in range(sizeofQ):
 
            # Remove the first word from the queue
            word = [j for j in Q.popleft()]
            #Q.pop()
 
            # For every character of the word
            for pos in range(wordlength):
                 
                # Retain the original character
                # at the current position
                orig_char = word[pos]
 
                # Replace the current character with
                # every possible lowercase alphabet
                for c in range(ord('a'), ord('z')):
                    word[pos] = chr(c)
 
                    # If the new word is equal
                    # to the target word
                    if ("".join(word) == target):
                        return level + 1
 
                    # Remove the word from the set
                    # if it is found in it
                    if ("".join(word) not in D):
                        continue
                         
                    del D["".join(word)]
 
                    # And push the newly generated word
                    # which will be a part of the chain
                    Q.append("".join(word))
 
                # Restore the original character
                # at the current position
                word[pos] = orig_char
 
    return 0
'''
Given two words (start and end), and a dictionary, find the shortest transformation sequence from start to end, such that:
Only one letter can be changed at a time
Each intermediate word must exist in the dictionary
If there are multiple such sequence of shortest length, return all of them.

Idea of the solution : We do normal BFS as is done for calculating the shortest path. We only take care of all the possible parents for a node which happens in following 2 cases :
1) Node x discovers node y and y is unvisited. x is parent of y.
2) Node x discovers node y and y is visited and distance[y] = distance[x] + 1.
Once we have constructed the parents, we do backtracking to construct all possible path combinations back from target to source.
Note that since we are constructing the reverse path, it might be helpful to swap start and end in the beginning.'''
def findladders(start,end,dict):
	if start == end:
		return [[start]]
	result = []
	dict = set(dict)
	queue = deque([(start,[start])])
	while queue:
		for _ in range(len(queue)):
			temp, path = queue.popleft()
			if temp == end:
				result.append(path)# pop all the extensions from prev step and add them one by one to the result if found that last elem matches the end
			for i in range(len(start)):
				for j in range(26):
					newtemp = temp[:i] + chr(j + ord("a")) + temp[i+1:]
					if newtemp in dict and newtemp not in path:
						queue.append(newtemp,path+[newtemp])# all possible combinations are found out and put in the deque, in the next iteration
						# each of them is extended one by one as popped from the queue
		if result:
			return result
	return []
'''
Given an N x M matrix A of non-negative integers representing the height of each unit cell in a continent, the "Blue lake"
 touches the left and top edges of the matrix and the "Red lake" touches the right and bottom edges.
Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.
Find the number of cells from where water can flow to both the Blue and Red lake.

Soln: Run bfs twice, one from all the co-ordinates connected to red lake and other from blue lake.
Mark the visited cell and count the cell which are visited in both bfs. We will solve the problem the reverse way, starting from thr lakes
Maintain a queue, initially append all the co-ordinates adjacent to blue lake. After that append all the cells adjacent to the current cell and have height >= height of current cell.
Mark the cell blue. Do similar with cells adjacent to red lake. Count the cells with both red and blue visited.'''
def redBlueCells(A):
	n,m = len(A), len(A[0])
	blue = [[False]*m for _ in range(n)]
	red = [[False]*m for _ in range(n)]
	def ourdfs(i,j,arr):
		arr[i][j] = True
		queue = deque([(i,j)])
		while queue:
			i,j = queue.popleft()
			for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
				if 0<= x < n and 0<=y<m and not arr[x][y] and A[x][y]>=A[i][j]:
					arr[x][y] = True
					queue.append((x,y))
	for i in range(n):
		ourdfs(i,0,red)
		ourdfs(i,m-1,blue)
	for j in range(m):
		ourdfs(0,j,red)
		ourdfs(n-1,j,blue)
	return sum([red[i][j] and blue[i][j] for i in range(n) for j in range(m)])
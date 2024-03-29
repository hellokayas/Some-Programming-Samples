# Programming ToolKit


## Tree(inorder,preorder,postorder,level-order), BFS, DFS, Topological Sort

General BFS code
```bash
visited = []
queue = [(x)] # initialized
visited.append(x)
while queue:
	s = queue.pop(0)
	# do whatever reqd with this
	for i in graph[s]:
		queue.append(i)
		visited.append(i)
```
O(V+E)
Now let us look at general DFS code
```bash
visited = []
def dfs(start):
    visited.add(start)
    for nbr in graph[start]:
        if nbr not in visited:
            dfs(nbr)
```
Now let us look at the level order traversal of a binary tree
```bash
def f(root):
    if not root: return []
    q = deque([root])
    res = []
    while q:
        size = len(q)
        level = []
        for _ in range(size):
            node = q.popleft()
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
            level.append(node.val)
        res.append(level)
    return res

```

Depth First Traversals: 
(a) Inorder (Left, Root, Right) 

Algorithm Inorder(tree)
   1. Traverse the left subtree, i.e., call Inorder(left-subtree)
   2. Visit the root.
   3. Traverse the right subtree, i.e., call Inorder(right-subtree)

(b) Preorder (Root, Left, Right)

Algorithm Preorder(tree)
   1. Visit the root.
   2. Traverse the left subtree, i.e., call Preorder(left-subtree)
   3. Traverse the right subtree, i.e., call Preorder(right-subtree) 

(c) Postorder (Left, Right, Root)

Algorithm Postorder(tree)
   1. Traverse the left subtree, i.e., call Postorder(left-subtree)
   2. Traverse the right subtree, i.e., call Postorder(right-subtree)
   3. Visit the root.

We will have to look at how to get one of these three given the other two -> three combinations -> number 106 Leetcode
Now Problems!

## Binary tree root + target, return all root-to-leaf paths which sum to target
output is a list of lists ans = [[],[],...]. Inside each list will be a root-to-leaf path. We run dfs for finding each solution. Whenever we reach the target, we immediately know that this is
a solution and append this list into ans. We keep track of the curr path at each dfs call and whenever the sum of vals in the curr path == target sum, we put the currpath inside ans.
```bash
def dfs(target,root,curr_path):
    if not root: return
    target = target - root.val
	if target == 0 and not root.left and not root.right:
		curr_path.append(root.val)
		ans.append(curr_path)
    dfs(target,root.left,curr_path+[root.val])
    dfs(target,root.right,curr_path+[root.right])
    return

```
## Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
This is just level order traversal and at each level we just want to see the last elem. There is another way that reqs less space. See prob 199

## Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
We just need to do a dfs whenever we see a 1 in the grid. This dfs runs as long we are on a connected component. Thus the number of times we run dfs is same as the num of connected components,
which is the number of islands. We can maintain a visited set or everytime we see 1, we just change it to some other symbol so that we dont run dfs on this again.
```bash
def dfs(grid,i,j):
    if (i >= rows or j >= cols or i <0 or j < 0 or grid[i][j] != "1"): return
    grid[i][j] = "f"
    dfs(grid,i-1,j)
    dfs(grid,i,j+1)
    dfs(grid,i+1,j)
    dfs(grid,i,j-1)
    return

```

## Number of Connected Components in an Undirected Graph
Create the graph given the edges. Then run dfs from one node, this will get the connected comp. The number of times we run dfs = number of conn comp
```bash
def dfs(node):
    if node in visited: return
    visited.add(node)
    for nbr in adjList[node]:
        if nbr not in visited:
            dfs(nbr)
visited = set()
count = 0
for i in range(n):
    if i not in visited:
        dfs(i)
        count += 1
return count
```



## There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. Return true if you can finish all courses. Otherwise, return false.

This is topological sort where the nodes are the courses and the directed edges are from bi to ai. If there is a top ordering then we return true. We remove edges after forming graph starting from  those vertices that have lowest indegree and these are the vertices in the lowest order an so on.
```bash
def f(n,req):
	indeg = [0]*n
	for v, _ in req:
		indeg[v] += 1
	# the graph has been built
	res = []
	q = [] # this is the reqd order of the vertices
	for i in range(n):
		if indeg[i] == 0:
			q.append(i)
	# the first layer of vertices have been added
	number = 0
	while q: # now going to the next layer
		u0 = q.pop(0)
		res.append(u0)
		number += 1
		for v,u in req:
			if u == u0:
				indeg[v] -= 1 # removing those edges
				if indeg[v] == 0: # ready for the curr layer
					q.append(v)
		return True if number == n else False

# res is the list of vertices that are topologically sorted
```

## Given the root of a complete binary tree, return the number of the nodes in the tree.
This is tricky with no general concept. look at https://leetcode.com/problems/count-complete-tree-nodes/

## Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
Just do inorder traversal and return the k-1 th elem from that list. But there is a followup: What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?
That's a design question, basically we're asked to implement a structure which contains a BST inside and optimises the following operations :
Insert,Delete,Find kth smallest
Seems like a database description, isn't it? Let's use here the same logic as for LRU cache design, and combine an indexing structure (we could keep BST here) with a double linked list.

## Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
We need to find tehe parents of all the nodes and then find the ancestral path from the two given nodes till the root. Then reverse the paths so they match from the start and then find the spot from where they first branch. This will be the LCS.
```bash
hashtable = {}
def dfs(node,parent):
    if not node: return
    hashtable[node] = parent
    dfs(node.left,node)
    dfs(node.right,node)

dfs(root,None) # now all the parents are known
p_ancestors = []
while p:
    p_ancestors.append(p)
    p = hashtable[p]
# do similarly for the other node q
pParent = p_ancestors[::-1]
# do same for q's list
for np,nq in zip(pParent,qParent):
    if np == nq: lca = np
return lca
```

## Given a graph with n nodes labeled 0 to n-1, and edges[i] = [ai,bi]. Say if this graph is a tree or not

Given the list of edges, we can create the graph in three ways, whichever comes handy whenever.
```bash
G = defaultdict(list)
for u,v in edges:
    G[u].append(v)
    G[v].append(u)

# OR we can do 

adj_list = [[] for _ in range(n)]
for A, B in edges:
    adj_list[A].append(B)
    adj_list[B].append(A)

# OR we can do

adjList = {i:[] for i in range(n)}
for edge in edges:
    A, B = edge
    adjList[A].append(B)
    adjList[B].append(A)
```
Now just check if there is a cycle or not, then immediately return False. Else now check if after running a dfs we have appended all the n nodes in the visited set.
```bash
def detect_cycle(node, visited, parent):
    visited.add(node)
    for child in G[node]:
        if(child == parent): continue
        if(child in visited or detect_cycle(child, visited, node)):   # Current node is now parent
                return True
    return False

if detect_cycle(0, visited, -1): return False
return len(visited) == n
```
## find leaves of binary tree
extremely tricky dfs
```bash
def f(root):
    def helper(root):
        if not root: return 0
        left = helper(root.left)
        right = helper(root.right)
        level = max(left,right) + 1
        d[level].append(root.val)
        return level
    d = collections.defaultdict(list)
    helper(root)
    return list(d.values())
```
Insertion and deletion from Binary search tree should be practiced: https://www.techiedelight.com/deletion-from-bst/

## finding a path in maze using dfs
```bash
def dfs(i, j):
    if [i, j] == destination: return True
    for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)): # the directions
        x, y = i, j
        while 0 <= x+dx < m and 0 <= y+dy < n and not maze[x+dx][y+dy]:
            x, y = x+dx, y+dy # this is a special case where we do not stop for every nbr but go on along the nbrs i.e. proceed in one of the directions as long as we can
        if (x,y) not in seen: 
            seen.add((x,y))
            if dfs(x,y): return True # if reached destination, then the True comes out from within the loop
    return False
```
## inorder successor. Should we look at the other order successors and predecessors??
There are two possible situations here : Node has a right child, and hence its successor is somewhere lower in the tree. To find the successor, go to the right once and then as many times to the left as you could.
if node.right:
    curr = node.right
    while curr.left:
        curr = curr.left
    return curr
Node has no right child, then its successor is somewhere upper in the tree. To find the successor, go up till the node that is left child of its parent. The answer is the parent. Beware that there could be no successor (= null successor) in such a situation
while node.parent and node == node.parent.right:
    node = node.parent
return node.parent

## Given the root of a binary tree, return the maximum width of the given tree.
Since we are checking level by level, very similar to level order traversal, while appending to queue, also append the columkn number, the colnumber - the head of the level is the curr width, we update the max width at each stage
```bash
max_width = 0
# queue of elements [(node, col_index)]
queue = deque()
queue.append((root, 0))

while queue:
    level_length = len(queue)
    _, level_head_index = queue[0]
    # iterate through the current level
    for _ in range(level_length):
        node, col_index = queue.popleft()
        # preparing for the next level
        if node.left:
            queue.append((node.left, 2 * col_index))
        if node.right:
            queue.append((node.right, 2 * col_index + 1))

    # calculate the length of the current level,
    #   by comparing the first and last col_index.
    max_width = max(max_width, col_index - level_head_index + 1)
```
## The area of an island is the number of cells with a value 1 in the island. Return the maximum area of an island in grid. If there is no island, return 0. You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Define DFS on a cell and this will find the connected component, while finding the component add up the  total area while doing dfs. Then just return max of running dfs from all the cells
```bash
seen = set()
def dfs(r,c):
    #seen = set()
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1 and (r,c) not in seen): return 0
    seen.add((r,c))
    return (dfs(r,c+1) + dfs(r,c-1) + dfs(r-1,c) + dfs(r+1,c) + 1)
```

## check if a graph is bipartite given the adjacency list of each node,  where graph[u] is an array of nodes that node u is adjacent to.
Keep a dict called color where nodes are keys and the colors are the numbers 0 or 1. We will always try to given different colors to nbrs through dfs and if we see that there is a loop, G is not bipartite
```bash
color = {}
for node in range(len(G)):
    if node not in color:
        color[node] = 0
        stack = [node]
        while stack:
            node = stack.pop()
            for nbr in G[node]:
                if nbr not in color:
                    stack.append(nbr)
                    color[nbr] = color[node]^1
                elif color[node] == color[nbr]: return False
return True
```
## find all possible paths from one node to another in a DAG. Vertices are number 0 to n-1. we are searching 0 -> n-1 all paths
define dfs on a node that goes through all paths and then just call dfs(0)
```bash
def dfs(v):
    if v == n-1: yield [v]
    yield from ([v] + ls for u in graph[v] for ls in dfs(u))
```
## all possible full binary tree == Catalan number

## Binary tree pruning: Given the root of a binary tree, return the same tree where every subtree (of the given tree) not containing a 1 has been removed. A subtree of a node node is node plus every node that is a descendant of node.
```bash
left = self.pruneTree(root.left)
root.left = left
right = self.pruneTree(root.right)
root.right = right
if (root.val == 0 and root.left == None and root.right == None): return None
return root
```

## UNION FIND : https://leetcode.com/problems/possible-bipartition/ 
The structure is coded here in this example. We will use it in other problems too

Memorize this by heart like learning a poem
```bash
class UnionFind:
    
    def __init__(self,n):
        self.parent = [i for i in range(n+1)]
        self.rank = [1 for i in range(n+1)]
        
    def find(self,x):
        p = self.parent[x]
        change = []
        while p!= self.parent[p]:
            change.append(p)
            p = self.parent[p]
        for node in change:
            self.parent[node] = p    
        return p
    
    def union(self,x,y):
        p_x = self.find(x)
        p_y = self.find(y)
        
        if p_x == p_y:
            return False
        if self.rank[p_x] > self.rank[p_y]:
            self.rank[p_x] += self.rank[p_y]
            self.parent[p_y] = p_x
        else:
            self.rank[p_y] += self.rank[p_x]
            self.parent[p_x] = p_y
        return True
```
## Now consider the problem: We want to split a group of n people (labeled from 1 to n) into two groups of any size. Each person may dislike some other people, and they should not go into the same group. Given the integer n and the array dislikes where dislikes[i] = [ai, bi] indicates that the person labeled ai does not like the person labeled bi, return true if it is possible to split everyone into two groups in this way.

uf = UnionFind(n) data structure is created and then a dislike graph d is created by using the dislikes as an edgelist.
```bash
for x in range(1,n+1):
    if d[x]:
        leader = d[x][0]
        for nbr in d[x]:
            if uf.find(nbr) == uf.find(x):
                return False
            uf.union(nbr,leader)
return True
```

## calculating the left and right height of bin tree for counting the nodes of complete binary tree
```bash
def calc_left(node):
    height = 0
    while node:
        height+=1
        node = node.left
    return height

left_height = calc_left(root)
right_height = calc_right(root)
if (left_height == right_height):
    return (1<<left_height)-1 #bitwise for 2^height-1

# otherwise, we are at the last level of the tree, we need to recurse the leafs

return 1 + self.countNodes(root.left) + self.countNodes(root.right)
```
## Surrounded regions : Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'. A region is captured by flipping all 'O's into 'X's in that surrounded region.

A ‘O’ is not replaced by ‘X’ if it lies in region that ends on a boundary. Traverse the given matrix and replace all ‘O’ with a special character ‘.‘
Traverse four edges of given matrix and replace(‘.‘, ‘O’) for every ‘.‘ on edges. The remaining ‘.‘ are the characters that indicate ‘O’s (in the original matrix) to be replaced by ‘X’. Traverse the matrix and replace all ‘-‘s with ‘X’s. We will do the whole thing inplace and so use a deque so that we can pull out from the left and push from the right.
```bash
queue = collections.deque([])
for r in range(len(board)):
    for c in range(len(board[0])):
        if (r in [0, len(board)-1] or c in [0, len(board[0])-1]) and board[r][c] == "O":
            queue.append((r, c))
while queue:
    r, c = queue.popleft()
    if 0<=r<len(board) and 0<=c<len(board[0]) and board[r][c] == "O":
        board[r][c] = "."
        queue.extend([(r-1, c),(r+1, c),(r, c-1),(r, c+1)])

for r in range(len(board)):
    for c in range(len(board[0])):
        if board[r][c] == "O":
            board[r][c] = "X"
        elif board[r][c] == ".":
            board[r][c] = "O"
```

## The diameter of a tree is the number of edges in the longest path in that tree. There is an undirected tree of n nodes labeled from 0 to n - 1. You are given a 2D array edges where edges.length == n - 1 and edges[i] = [ai, bi]

Create adj list from the edgelist, i.e adj = defaultdict(list). Now start a BFS while keeping track of parents starting from 0. We need to keep track of parent so that we dont travel in the same direction in the tree as we came from. At the end of BFS, say we reach node u. STart andother BFS from u. Keep track of parents here to ensure that we are travelling in a new direction. Now calculate the number of steps we take in new direction till the BFS ends. This is the diameter.
```bash
# first BFS
q = deque()
q.append((0, -1)) # (vertex, parent)
while len(q) > 0:
    u, parent = q.popleft()
    for v in adj[u]:
        if v != parent:
            q.append((v, u))

# second BFS
d = 0
q.append((u, -1))
while len(q) > 0:
    sz = len(q)
    for i in range(sz):# every time this loop breaks we have gone another layer of bfs
        u, parent = q.popleft()
        for v in adj[u]:
            if v != parent:
                q.append((v, u))
    d += 1
# minus 1 because the last level is empty
return d - 1   
```

## array of strings equations given "xi==yi" or "xi!=yi". Return true if it is possible to assign integers to variable names so as to satisfy all the given equations, or false otherwise.

Create graph = defaultdict(list) and fill it up from the edge list which is the eqn list. edge if there is "=", else no edge. Then define DFS as usual defined above. 
```bash
for eqn in equations:
    visited = set()
    if eqn[1] == "!":
        dfs(eqn[0])
        if eqn[3] in visited:
            return False
return True
```

## For a binary tree T, we can define a flip operation as follows: choose any node, and swap the left and right child subtrees. A binary tree X is flip equivalent to a binary tree Y if and only if we can make X equal to Y after some number of flip operations. Given the roots of two binary trees root1 and root2, return true if the two trees are flip equivalent or false otherwise.

Trivial cases are 
if root1 == None and root2 == None: return True
if root1 == None and root2 != None: return False
if root1 != None and root2 == None: return False
If the roots match, return (self.flipEquiv(root1.left,root2.left) and self.flipEquiv(root1.right,root2.right)) or (self.flipEquiv(root1.left,root2.right) and self.flipEquiv(root1.right,root2.left))

## In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square [0, 0]. Return the minimum number of steps needed to move the knight to the square [x, y]. It is guaranteed the answer exists.

Two things to keep in mind here:
1. knight moves = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
2. When we are doing BFS on a grid to find the shortest path, it is good to consider the number of steps as a parameter inside the BFS and pass on to the next call of recursion
```bash
q = collections.deque([(0, 0, 0)])
while q:
    i,j,steps = q.popleft()
    if i == x and j == y:
        return steps + res
    for di,dj in moves:
        if (x - i) * di > 0 or (y - j) * dj > 0: # move towards (x, y) at least in one direction
            q.append((i + di, j + dj, steps + 1))
```
Now since it was an infinite chess board we need to do some tricks in order to prevent TLE. No biggie!

## Given the root of a binary tree, find the maximum value v for which there exist different nodes a and b where v = |a.val - b.val| and a is an ancestor of b. A node a is an ancestor of b if either: any child of a is equal to b or any child of a is an ancestor of b.

The idea is that all such a,b pairs will be on some path that connects root to leaf. So get all such root to leaf paths and store all paths in res. We already know how to do this.
def printRootToLeafPaths(node, path):

    # base case
    if node is None:
        return

    # include the current node to the path
    path.append(node.val)

    # if a leaf node is found, print the path
    if node.left is None and node.right is None:
        res.append(list(path))

    # recur for the left and right subtree
    printRootToLeafPaths(node.left, path)
    printRootToLeafPaths(node.right, path)

    # backtrack: remove the current node after the left, and right subtree are done
    path.pop()
    return res
Then just do the following:

ans = [max(i)-min(i) for i in res]
return max(ans)

## You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei: course prevCoursei has to be taken before course nextCoursei. In one semester, you can take any number of courses as long as you have taken all the prerequisites in the previous semester for the courses you are taking. Return the minimum number of semesters needed to take all courses. If there is no way to take all the courses, return -1

Set up the graph for the topological sort and then set up the queue as reqd. Then do top sort but keep track of two parameters count = 0, visitedcount = 0
```bash
while q:
    count += 1
    nexq = []
    for node in q:
        visitedcount += 1
        endnodes = graph[node]
        for endnode in endnodes:
            indegree[endnode] -= 1
            if indegree[endnode] == 0: # all prereq learnt
                nexq.append(endnode)
    q = nexq
    return count if visitedcount == n else -1
```

## Given a root of an N-ary tree, you need to compute the length of the diameter of the tree. The diameter of an N-ary tree is the length of the longest path between any two nodes in the tree. This path may or may not pass through the root.

This is very hard. Look at various solutions. This is tricky and need to be memorized. https://leetcode.com/problems/diameter-of-n-ary-tree/solution/

## Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1. A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that: All the visited cells of the path are 0. All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner). The length of a clear path is the number of visited cells of this path.

directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
Define what a nbr of a (row,col) look like by giving a func called getnbr.
```bash
 def get_neighbours(row, col):
    for row_difference, col_difference in directions:
        new_row = row + row_difference
        new_col = col + col_difference
        if not(0 <= new_row <= max_row and 0 <= new_col <= max_col):
            continue
        if grid[new_row][new_col] != 0:
            continue
        yield (new_row, new_col)
```
Now we just run a bfs to find the path. Store the distance covered this far inside the grid at that cell where we are right now.
```bash
while queue:
    row, col = queue.popleft() # this means we ar using a deque
    distance = grid[row][col]
    if (row, col) == (max_row, max_col):
        return distance
    for neighbour_row, neighbour_col in get_neighbours(row, col):
        grid[neighbour_row][neighbour_col] = distance + 1
        queue.append((neighbour_row, neighbour_col))

# There was no path.
return -1
```

## You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of size rows x columns, where heights[row][col] represents the height of cell (row, col). You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or right, and you wish to find a route that requires the minimum effort. A route's effort is the maximum absolute difference in heights between two consecutive cells of the route. Return the minimum effort required to travel from the top-left cell to the bottom-right cell.

We will do binary search on possible range of values. Calculate f(mid) and move to the left or right suitably. We need to design f here.

```bash
def f(mid):
    visited = [[False]*col for _ in range(row)]
    queue = [(0, 0)]  # x, y
    while queue:
        x, y = queue.pop(0)
        if x == row-1 and y == col-1:
            return True
        #visited[x][y] = True
        for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
            adjacent_x = x + dx
            adjacent_y = y + dy
            if 0 <= adjacent_x < row and 0 <= adjacent_y < col and not visited[adjacent_x][adjacent_y]:
                current_difference = abs(heights[adjacent_x][adjacent_y]-heights[x][y])
                if current_difference <= mid:
                    visited[adjacent_x][adjacent_y] = True
                    queue.append((adjacent_x, adjacent_y))
```

## Given an m x n integer matrix grid, return the maximum score of a path starting at (0, 0) and ending at (m - 1, n - 1) moving in the 4 cardinal directions. The score of a path is the minimum value in that path.

this will be similar to the prev problem, we have done dfs which outputs T/F based on whether we can reach the last cell starting from i,j with min val in any path as mid. Here f(mid) = dfs(i,j,mid,grid[-1][-1]) Then normal bin search
while start < end:
    mid = (start + end) // 2
    visited = [[False] * n for _ in range(m)]
    if dfs(0, 0, mid, visited):
        start = mid + 1 # current is working, move to one step right
    else:
        end = mid
        
return end - 1 # or start - 1

```bash
def dfs(i, j, mini, visited):
    if i == m - 1 and j == n - 1:
        return True
    visited[i][j] = True
    for dx, dy in directions:
        nx, ny = i + dx, j + dy
        if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
            if grid[nx][ny] >= mini:
                if dfs(nx, ny, mini, visited): return True
    return False
```

## Given the root of a binary tree, each node in the tree has a distinct value. After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees). Return the roots of the trees in the remaining forest. You may return the result in any order.

Just recursively call the func so that it deletes the nodes to be deleted and then just call this function on the root. BUt handle the case where root is not to be deleted.

```bash
def helper(node):
    if not node:
        return None
    node.left = helper(node.left)
    node.right = helper(node.right)
    
    # add children of a node that is to be deleted
    if node.val in to_delete:
        if node.left: 
            ans.append(node.left)
        if node.right:
            ans.append(node.right)
        return None
    return node
```
Now call helper on root. but root might not be something to be deleted, so just add this to the result.

## https://leetcode.com/problems/populating-next-right-pointers-in-each-node/ 
no concept, just manipulate in a tricky way

## one of the hardest problems : ou are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water. An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other. Return the number of distinct islands. 

The official solution is the best explanation here. read app1 for intuition followed by app3 for the optimal soln which is hashing by path signature. The code is clear. We do a DFS that stores the path sign while traversing the arr. Then the same idea of calling dfs and getting the number of connected components which is the num of islands but two islands can have same path sign and hence will counted only once.

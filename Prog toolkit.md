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
def dfs(v):
	visited.add(v)
	# do whatever reqd with v
	for nbr in graph[v];
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


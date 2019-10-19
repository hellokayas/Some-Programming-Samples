def wordcount(l):
	ans = {}
	s = l.split()
	for w in s:
		ans[w] = ans.get(w,0) + 1

	print(sorted(list(ans.items())))

ls = ''' This is some sentence. With whitespace	like tabs and
newline as well to illustrate how this function wordcount
works. This is just to make sure that some words repeat '''
		
wordcount(ls)

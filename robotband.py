def count(l1,l2,l3,l4,val):
     count = 0
     for i in l1:
               for j in l2:
                    for k in l3:
                         for m in l4:
                              ans = i ^ j ^ k ^ m
                              if ans == val:
                                   count  = count +1
     return count



 
                    

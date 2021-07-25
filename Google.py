# Intern ph interview
'''
You are given a sorted list of distinct integers from 0 to 99, for instance [0, 1, 2, 50, 52, 75]. Your task is to produce a string that describes numbers missing from the list; in this case "3-49,51,53-74,76-99".

Examples:

[] “0-99”
[0] “1-99”
[3, 5] “0-2,4,6-99”
'''

def missing(l):
    ...:     out = ""
    ...:     for i in range(len(l)):
    ...:         if i < len(l)-1:
    ...:             if l[i+1]-l[i] > 2:
    ...:                 out += str(l[i]+1) + "-" + str(l[i+1]-1) + ","
    ...:             elif l[i+1]-l[i] > 1:
    ...:                 out += str(l[i]-1) + ','
    ...:         else:
    ...:             if l[i] < 99:
    ...:                 if 99-l[i] > 2:
    ...:                     out += str(l[i]+1) + "-99"
    ...:                 elif 99-l[i] > 1:
    ...:                     out += "99"
    ...:     return out

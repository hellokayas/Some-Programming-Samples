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

    
   '''
   Given a string (1-d array) , find if there is any sub-sequence that repeats itself.
Here, sub-sequence can be a non-contiguous pattern, with the same relative order.
This problem can be solved using the DP for longest repeating subseq prob and check if the longest len value > 0.

This solution works O(n) time and O(n) memory. It can be trimmed to use O(1) memory as well. The trick is that if you remove the non repeated characters, the remaining should be palindrome (if it is not then for sure there is a repeated). Also even if it is palindrome with odd length you have to check that the middle letter is not different from its left or right.

You can count the letter and check for palndrom in O(n).

Step 1: Scan the string to count the number of each letter.
(mean while if the count of any letter is 4 or more, we already find a repetition of two character, the same character followed by itself and return YES)
If no character is repeated, return NO.

Step 2: Scan second time and append each letter that has been repeated (using the count array from the first step) to a new string (let say a StringBuilder)

Step 3: If the new string is not palindrom, return YES.
else if newString.length() is odd and the middle is different from one previous return YES.

return NO.
Eg:

1. abab <------yes, ab is repeated
2. abba <---- No, a and b follow different order
3. acbdaghfb <-------- yes there is a followed by b at two places
4. abcdacb <----- yes a followed by b twice

The above should be applicable to ANY TWO (or every two) characters in the string and optimum over time.'''

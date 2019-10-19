import sys
import time

print ("processing...\\")
syms = ['\\', '|', '/', '-']
bs = '\b'

for _ in range(10):
    for sym in syms:
        sys.stdout.write("\b%s" % sym)
        sys.stdout.flush()
        time.sleep(1)

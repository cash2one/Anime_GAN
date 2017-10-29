import os
from multiprocessing import Queue
def f():
    return 1
def f2(a):
    a.append(1)
def f1(a,b,c):
    return a,b,c
class a():
    b=None
Ns = a()
Ns.b = [1,2,3,4]
f2(Ns.b)
print(Ns.b)
#while True:
 #   print('Queue : {}'.format(q.get()))
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:15:39 2023

@author: User
"""

#Alister 2023-02-26 Start implement multiprocessing
from multiprocessing import Process, Queue
import time
global c1
global c2
import os
sleep_sec = 0.50
def Get_Frame(q1,):
    global c
    c1=1
    k=1
    print('[Get_Frame] module name:', __name__)
    print('[Get_Frame] parent process:', os.getppid())
    print('[Get_Frame] process id:', os.getpid())
    while k<8:
        #t = time.time()
        q1.put(c1)
        #print("[Get_Frame] {}".format(q1.qsize))
        print("[Get_Frame] q1.put():{}".format(c1))
        c1+=1
        k+=1
    
        time.sleep(sleep_sec)
        
    #return True

def Model_Inference(q1,q2):
    global c2
    c2=1
    j=1
    print('[Model_Inference] module name:', __name__)
    print('[Model_Inference] parent process:', os.getppid())
    print('[Model_Inference] process id:', os.getpid())
    while j<4:
        c1 = q1.get()
        #t2 = time.time()
        q2.put(c2)
        #print("[Model_Inference] {}".format(q2.qsize))
        print("[Model_Inference] q1.get():{}".format(c1))
        print("[Model_Inference] q2.put():{}".format(c2))
    
        c2+=1
        j+=1
        time.sleep(sleep_sec)
        
    #return True

def PostProcess(q2):
    l=1
    
    print('[PostProcess] module name:', __name__)
    print('[PostProcess] parent process:', os.getppid())
    print('[PostProcess] process id:', os.getpid())
    while l<4:
        c2 = q2.get()
        l+=1
        print("[PostProcess] q2.get():{}".format(c2))
        time.sleep(sleep_sec)
    #return True


if __name__ == '__main__':
    
    q1 = Queue()
    q2 = Queue()
    
    p1 = Process(target=Get_Frame,args=(q1,))
    p1.start()
    
    cnt = 0
    
    '''
    while not q1.empty():
        print('q1 no: ', cnt, ' ', q1.get())
        cnt += 1
    '''
   
    #print("q1 : {}".format(q1.get()))
    
    p2 = Process(target=Model_Inference,args=(q1,q2,))
    p2.start()
    
    '''
    while not q2.empty():
        print('q2 no: ', cnt, ' ', q2.get())
        cnt += 1
    '''
    
    p3 = Process(target=PostProcess,args=(q2,))
    p3.start()
    
    #print("q2 :{} ".format(q2.get()))
    
    p1.join()
    p2.join()
    p3.join()
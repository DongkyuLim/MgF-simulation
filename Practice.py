import os
import multiprocessing as mp
from multiprocessing import Pool
import threading
import time
import datetime
from scipy import integrate
import numpy as np

 
def multiprocess():
    print("non multiprocess")

     #멀티 프로세싱을 위한 CPU 숫자 확인 및 풀 만들기
    num_cores = mp.cpu_count()
    pool = Pool(num_cores)
    start = int(time.time())
    ojbect_list = []
    for i in range(1,10):
        ojbect_list.append(i)

    #멀티 프로세싱 워커 호출
    pool.map(work_func, ojbect_list)
    end = int(time.time())

    #메모리 릭 방지 위해 사용
    pool.close()
    pool.join()

    print("Number of Core : " + str(mp.cpu_count()))
    print("***run time(sec) : ", end-start)   

def work_func(i,j):
    start_point = 2**i
    end_point = 2**j
    result = integrate.quad(lambda x : x**2, start_point,end_point)
    print(result)
    return result


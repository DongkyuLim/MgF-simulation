import parmap
import numpy as np
import multiprocessing

num_cores = multiprocessing.cpu_count()-1

def square(x):
    return x**2


data = list(range(1,25))

if __name__ == '__main__':
    result = parmap.map(square,data, pm_processes=num_cores)    
    print(result)
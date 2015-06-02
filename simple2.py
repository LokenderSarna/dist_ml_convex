from mpi4py import MPI
import numpy as np
import math, sys

def main():
    
    # Define main execute
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #how do we create an object for access like this, but not in this file?????
    simpleObj = SimpleObject(5,6)
    
    
    # Now we decide wether this is a master or worker call of this script
    if rank == 0:
        data = [(i+1)**2 for i in range(size)]
        
        # just add a special case that is dependant on the object
        if size > 5:
            data[5] = simpleObj.x
            
            # We would need to access an instance of simple object like this:
            # data[5] = simpleObjectInstance.x
    
    else:
        data = None
    
    data = comm.scatter(data, root=0)
    print "I have rank = %d and have data = %d" %(rank, data)


class SimpleObject:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y 


if __name__=="__main__":
    main()
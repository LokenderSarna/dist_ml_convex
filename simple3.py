from mpi4py import MPI
import numpy as np
import math, sys, inspect

def main():
    size = 3
    array = [SimpleObject(lambda x : x + 10), SimpleObject(lambda x : x + 20), SimpleObject(lambda x : x + 30)]
    #array.append(SimpleObject(lambda x : x + 10))
    #array.append(SimpleObject(lambda x : x + 20))
    #array.append(SimpleObject(lambda x : x + 30))
    print "hola"
    ds = DoSomething(array)
    ds.execute()

    

# Mini version of our oracle
class SimpleObject:
    
    def __init__(self, f):
        self.f = f

# Mini version of our gradient descent
class DoSomething:
    
    def __init__(self, objarray):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        self.comm = comm
        self.rank = rank
        self.size = size
        if self.rank > 0:
            self.obj = objarray[rank]
    
    
    def execute(self):
        # Now we decide wether this is a master or worker call of this script
        if self.rank == 0:
            data = np.random.random(1)
            print "I am the master and the random number is %f" %data
            
    
        else:
            data = None
    
        data = self.comm.bcast(data, root=0)
        
        if self.rank > 0:
            result = self.obj.f(data)
            print "I have rank = %d. I received data = %f, and computed result %f" %(self.rank, data, result)
        

if __name__=="__main__":
    main()
    
    # If I were to do this there is no way for the parameters to be sent
    #simpleObj = SimpleObject(5,6)
    #SimpleObject.execute(simpleObj)
from mpi4py import MPI
import numpy as np
import math, sys, inspect, itertools

#define the needed variables
comm = MPI.COMM_WORLD
#comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()

print "hello my rank is %s" % rank

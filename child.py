from mpi4py import MPI
import numpy as np
import math, sys, inspect
from dist_ml_convex import *

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

comm.Bcast(convex_problem, root=0)

x = convex_problem[0]
oracle = convex_problem[1][rank]
grad_f = oracle.grad_f
new_x = grad_f(x)

comm.Reduce([new_x, MPI.DOUBLE], None, op=MPI.SUM, root=0)
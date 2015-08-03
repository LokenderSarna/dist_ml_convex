from dist_ml_convex import *

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
index, edges = [0], []
for i in range(size):
    pos = index[-1]
    index.append(pos+2)
    edges.append((i-1)%size)
    edges.append((i+1)%size)   
topo = comm.Create_graph(index[1:], edges)
if rank == 3:
    print edges
    print topo.Get_neighbors(rank)
    print topo.Get_topo()
    print topo.Get_dims()
    print index



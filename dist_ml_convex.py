from mpi4py import MPI
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math, sys, time, csv

# Defining a layer of abstraction for each fi. Which will likely end up being a cut where each fi is responsible for some subset of data a machine learning algorithm needs to sift through
class FirstOrderOracle:

    # Where f is the function and grad_f is its gradient, f_dim
    def __init__(self, grad_f, dim_f, f=None):
        if f is None:
            f = None
        else:
            self.f = f
        self.grad_f = grad_f
        self.dim_f = dim_f

# Defaults for gradient descent parameters
EPSILON_DEFAULT = 0.001
MAX_ITER_DEFAULT = 10000
ALPHA_DEFAULT = 0.001

# A drawing graph function from: https://www.udacity.com/wiki/creating-network-graphs-with-python
def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    # edge_labels = dict(zip(graph, labels))
    # nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 # label_pos=edge_text_pos)

    # show graph
    plt.show()      

# Here we will define a gradient descent method that uses a gossip based implementation, references oracles as graph nodes.
class GradientDescentGossip:
    
    # Starting this off simple with default parameters for alpha, epsilon and max_iter.
    def __init__(self, oracles, alpha=None, epsilon=None, max_iter=None, x_init=None, lipschitz=None):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.oracle = oracles[self.rank]
        self.dim_f = self.oracle.dim_f
        self.grad_f = self.oracle.grad_f
        
        # Define whether or not to use the lipschitz values for decreasing alpha
        if lipschitz is True:
            self.lipschitz = True
        else:
            self.lipschitz = False
        
        # Since alpha will be a function our default will also be a function to match type checking
        if alpha is None:
            self.alpha = (lambda x : ALPHA_DEFAULT)
        else:
            self.alpha = alpha
        
        # Assigning epsilon value which is the size of our solution space
        if epsilon is None:
            self.epsilon = EPSILON_DEFAULT
        else:
            self.epsilon = epsilon
        
        # The maximum number of iterations before the algorithm terminate without a solution in the epsilon space
        if max_iter is None:
            self.max_iter = MAX_ITER_DEFAULT
        else:
            self.max_iter = max_iter
        
        # We will randomly choose our x initial values for gradient descent, the x values being the values we are optimizing
        if x_init is None:
            if self.rank == 0:
                data_bcast = np.random.random((self.dim_f, 1))
            else:
                data_bcast = None
            # There might be a higher communication cost of using a randomly initialized x vector, but for now lets leave it
            data_bcast = self.comm.bcast(data_bcast, root=0)     
            self.x = data_bcast
            self.x_previous = np.zeros((self.dim_f, 1))
        else:
            self.x = x_init
            self.x_previous = np.zeros((self.dim_f, 1))
        
        # Check statement to see if the user has inputed the correct number of processes
        if self.size != len(oracles) and self.rank == 0:
            # Note later you should add a real error to be triggered
            print "ERROR: For the gossip implementation your script call should be in the form:"
            print "mpirun -np <int: number of oracles> python <string: name of script>"
            
        # Now lets do the computation to define our graph. Since we are using probabilities to define edges, we will need to define the graph in one place so lets do it in the rank == 0 process.
        if self.rank == 0:
            
            # Where "np.sqrt( (2*np.log(self.size)) / self.size )" defines the edge probability that leads to a good graph
            edge_exists = lambda size : True if np.random.random() < np.sqrt( (2*np.log(size)) / size ) else False
            
            # Initialize index and edges matrices. Also a graph matrix for display
            index, edges, graph, degrees = [], [], [], np.zeros(16)
            
            # Now lets fill our edge and index matrices, Note the index_count variables allows us to track the pointers from our index array to our edges array.
            index_count = 0
            for ref_process in range(len(oracles)):               
                for current_process in range(len(oracles)):
                    # If our probability function says we should place an edge between processes, we do so.
                    if ref_process < current_process and edge_exists(self.size):
                        index_count += 1
                        edges.append(current_process)
                        graph.append((ref_process, current_process))
                        degrees[ref_process] += 1
                        degrees[current_process] += 1
                # Now update the index array
                index.append(index_count)
            
            # Initialize zero array for weights
            weights = np.zeros((self.size, self.size))
            
            # Now we need to find the weights associated with each edge.
            for edge in graph:
                edge1, edge2 = edge[0], edge[1]
                weights[edge1, edge2] = weights[edge2, edge1] = 1 / max(degrees[edge1], degrees[edge2])
            # Assign self weights, that is the diagonal of the matrix
            for ref_node in range(self.size):
                sum_of_weights = 0
                for current_node in range(self.size):
                    sum_of_weights += weights[ref_node][current_node]
                weights[ref_node][ref_node] = 1 - sum_of_weights
            
            np.set_printoptions(suppress=True)
            np.set_printoptions(precision=2)            
            
            # Show the created graph via networkx
            # draw_graph(graph)
            
            # Now that we have defined our graph, weights, etc. We need to notify the other processes and give them the info. 
            data_bcast = [index, edges, graph, weights]
        
        # If rank is not equal to 0, ie all other processes
        else:
            data_bcast = None
        
        # There might be a higher communication cost of using a randomly initialized x vector, but for now lets leave it
        data_bcast = self.comm.bcast(data_bcast, root=0)
        
        # Assign and seperate the broadcasted arrays
        index, edges, graph, weights = data_bcast[0], data_bcast[1], data_bcast[2], data_bcast[3]
        
        neighbours = []
        # Now generate the nieghbours array so each process knows who to send x updates to.
        for node in range(len(weights[self.rank])):
            if weights[self.rank][node] != 0 and self.rank != node:
                neighbours += [node]
        # Now assign to the process instance for reference in gradient descent execution.
        self.neighbours = neighbours
        # Just assign the weights the process needs to know.
        self.weights = weights[self.rank]
        
        
    # Now we perform the execution of gradient descent
    def execute(self):
        
        cont_iter = True
        num_iter = 0
        
        # Begin the iteration steps to update our weights of self.x
        while cont_iter == True:
            
            num_iter += 1
            # So for this first implementation we are just going to use iteration limit for a termination criteria
            if num_iter > self.max_iter:
                # If we can't find a value within the max iteration limit stop the execution
                print "Stopping after reaching iteration limit. Here was the final weights found at rank = %d:" %self.rank
                print self.x
                cont_iter = False
                continue
            
            # Compute grad at this process with its current x value
            grad_f_x = self.grad_f(self.x)
            
            # Send the value found to neighbours
            for neighbour in self.neighbours:
                self.comm.Send([self.x, MPI.DOUBLE], dest=neighbour)
            
            # Recieve the x values the other processes found on this iteration, I am assigning grad_f_x_received to grad_f_x just as a placeholder for incoming receptions. They match types so the initialization is easy.
            grad_f_x_received = grad_f_x
            # sum_grads is the sum of gradients normalized and weighted via our initialized weights matrix and received values for grad_f_x_j
            sum_grads = 0
            
            for neighbour in self.neighbours:
                self.comm.Recv([grad_f_x_received, MPI.DOUBLE], source=neighbour)
                # Here we need to fulfill the equation x_i_k+1 = x_i_k - alpha_k (sum of all neighbours j) weight_i_j*grad_f_j(x_j)
                sum_grads += self.weights[neighbour]*grad_f_x_received
            
            # Update the x value based on a weighted summation of grad_fs
            self.x = self.x - self.alpha(num_iter)*sum_grads
            
            if self.rank == 12 and num_iter % 1000 == 0:
                print self.x
            

# Gradient_descent will take a convex problem as input and find the optimal x* 
class GradientDescent:
    
    # Parameters taken as input; alpha: the step size function, epsilon the stopping critera, epsilon: the stop criterion, max_iter: upper limit on the number of iterations
    def __init__(self, oracles, alpha=None, epsilon=None, max_iter=None, x_init=None, lipschitz=None):
        
        # Here our gradient descent initialization will need to be assigned as the master
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.oracles = oracles
        self.dim_f = self.oracles[0].dim_f
        
        # Define whether or not to use the lipschitz values for decreasing alpha
        if lipschitz is True:
            self.lipschitz = True
        else:
            self.lipschitz = False
        
        # Since alpha will be a function our default will also be a function to match type checking
        if alpha is None:
            self.alpha = (lambda x : ALPHA_DEFAULT)
        else:
            self.alpha = alpha
        
        # Assigning epsilon value which is the size of our solution space
        if epsilon is None:
            self.epsilon = EPSILON_DEFAULT
        else:
            self.epsilon = epsilon
        
        # The maximum number of iterations before the algorithm terminate without a solution in the epsilon space
        if max_iter is None:
            self.max_iter = MAX_ITER_DEFAULT
        else:
            self.max_iter = max_iter
        
        # We will randomly choose our x initial values for gradient descent, the x values being the values we are optimizing
        if x_init is None:
            if self.rank == 0:
                data_bcast = np.random.random((self.dim_f, 1))
            else:
                data_bcast = None
            # There might be a higher communication cost of using a randomly initialized x vector, but for now lets leave it
            data_bcast = self.comm.bcast(data_bcast, root=0)     
            self.x = data_bcast
            self.x_previous = np.zeros((self.dim_f, 1))
        else:
            self.x = x_init
            self.x_previous = np.zeros((self.dim_f, 1))
            
        # Just to initialize the values I'm going to assign the current g_x and the previous g_x as the current x
        self.g_x = self.x
        self.g_x_previous = self.x
        self.L = 0
        
        # Now if lipschitz has been switched on we need to define the alpha to be a function of L
        if self.lipschitz == True:
            self.alpha = (lambda iteration : 1 / (self.L * np.sqrt(iteration)))
                
        # Check statement to see if the user has inputed the correct number of processes
        if self.size != len(self.oracles) + 1 and self.rank == 0:
            # Note later you should add a real error to be triggered
            print "ERROR: Your script call should be in the form:"
            print "mpirun -np <int: number of oracles + 1> python <string: name of script>"
        
        # Now if the process call being made here is of rank above 0 we know it is a child and needs to be assigned the correct oracle with associated grad_f
        if self.rank > 0:
            # Note that we are assigned the oracles at the -1 index to account for the master at index 0
            self.oracle = oracles[self.rank - 1]
        
    
    # Now we need our method to start the process calls
    def execute(self):
        
        num_iter = 0
        cont_iter = True
        sol_found = False      
        
        # First we will seperate the worker and master executions
        if self.rank == 0:
            
            # Here we initiate the loop to find an x solution to the gradient descent
            while cont_iter == True:
                num_iter += 1
                
                # Now we need to either send a self.x vector or flag that child processes should finish their executions
                if num_iter > self.max_iter or sol_found == True:
                    # If we can't find a value within the max iteration limit stop the execution
                    if num_iter > self.max_iter:
                        print "Stopping after reaching iteration limit. Here was the final weights found:"
                        print self.x
                    else:
                        print "Gradient descent has found a solution:"
                        print self.x
                        
                    cont_iter = False
                    data_bcast = cont_iter
                else:
                    # Send the current x vector to the children
                    data_bcast = self.x
                                
                # Initialize our masters summation of the gradients
                g_x = np.zeros_like(self.x)
                        
                # Now broadcast our self.x value so the other worker processes can compute their oracles gradient
                data_bcast = self.comm.bcast(data_bcast, root=0)
                
                # Here we need to check that data_bcast just sent wasn't a flag to stop the worker execution. If it was then we need to stop our master execution
                if cont_iter == False:
                    if sol_found == True:
                        return self.x
                    else:
                        return None
        
                x_out = np.zeros_like(self.x)
                # Perform the reduce to find the g_x sum of gradient vector   
                self.comm.Reduce([x_out, MPI.DOUBLE], [g_x, MPI.DOUBLE], op=MPI.SUM, root=0)    
                
                if self.lipschitz == True:
                    # Now update the lip value if needed
                    self.g_x_previous = self.g_x
                    self.g_x = g_x
                    
                    # Now if we are on the first iteration we need to just use the current values and no previous ones
                    if num_iter == 1:
                        self.L = self.g_x / self.x
                    # Otherwise use the difference between the gradients to find the lipschitz constant
                    else:
                        self.L = np.linalg.norm(self.g_x - self.g_x_previous) / np.linalg.norm(self.x - self.x_previous)
                    # Now we do book keeping on the previous x
                    self.x_previous = self.x
                    
                    
                # Here we do the update if we aren't within the epsilon value
                if np.linalg.norm(g_x) > self.epsilon:                        
                    # Note that fi is a function that goes from fi: R^d -> R^d, xk+1 = xk - alpha(num_iter)*f(xk)
                    self.x = np.subtract(self.x, np.multiply(self.alpha(num_iter), g_x))
                    
                    if num_iter%1000 == 0:
                        print "The current self.x value is:"
                        print self.x
                        print "np.linalg.norm(g_x) = %f" %np.linalg.norm(g_x)
                    
                else:
                    sol_found = True
        
        # Worker execution is as follows
        else:
            
            # Here we initiate the loop to find an x solution to the gradient descent
            while cont_iter == True:
        
                # For our worker execution case these variables do not need to send values and are thus assigned to None
                data_bcast = None
                g_x = None
        
                # Now broadcast our self.x value so the other worker processes can compute their oracles gradient
                data_bcast = self.comm.bcast(data_bcast, root=0)
        
                # Check to make sure we haven't been flagged to stop worker execution
                if isinstance(data_bcast, bool):
                    cont_iter = False
                    continue
                
                x_out = np.zeros_like(self.x)
                # Now if the rank is above 0 the process is the child and should compute the gradient at current x
                x_out = self.oracle.grad_f(data_bcast)
                
            
                # Perform the reduce to find the g_x sum of gradient vector
                self.comm.Reduce([x_out, MPI.DOUBLE], [g_x, MPI.DOUBLE], op=MPI.SUM, root=0)
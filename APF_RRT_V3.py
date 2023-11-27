import math
import numpy as np
from test_2 import *
from utils import *
from scipy import spatial
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import matplotlib.animation as animation




#constants
MAXITER=15000
HEIGHT=60
WIDTH=60



class AFP_final:
    ''' Potential Field based Extention procedure for Sampling ALgorithms'''

    def __init__(self):
        self.obs_rectangle2 = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2],
            [0,30,15,8],
            [50,40,5,3],
            [40,40,3,10],
            [30,35,3,10],
            [40,25,10,3],
            [20,60,20,20],
            [80,60,10,10],
            [60,60,10,10],
            [70,80,10,10],
            [80,0,10,50],
            [60,80,10,19],
            [30,25,5,5],
            [50,57,7,10]
        ]

        self.obs_boundary = [
            [0, 0, 1, 103],
            [0, 103, 103, 1],
            [1, 0, 103, 1],
            [103, 1, 1, 103]
        ]



        '''List of rectangular obstacles in the format [[ox,oy,w,h],[].[]'''
        self.obs_centroid=get_centroid(self.obs_rectangle2)
        self.obs_vertix=get_obs_vertex(self.obs_rectangle2)
        self.v_p=vertex_2_point(self.obs_vertix)
        self.tree = spatial.cKDTree(self.v_p)
        self.goal_sample_rate=0.05
        #print(" \n \n \n \n vertix point : \n \n \n",self.v_p,"\n \n \n \n")
        #print(self.obs_centroid,"Obstacle centroid \n \n \n")
        #print("Obs vertices: \n \n \n", self.obs_vertix)
        
        #Max no of iterations
        self.iter_max = 10000


        self.start_pos=[3,7]
        self.end_pos=[90,97]
        self.x_range=(0,100)
        '''Map Dimesnions'''
        self.y_range=(0,100)
        '''Map Dimenisons'''

        self.start_node=Node(self.start_pos)
        self.end_node=Node(self.end_pos)
        self.delta = 0.5
        '''Obstacle clearance. Similar to dstar. will put it to zero'''

        self.vertex = [self.start_node]
        self.step_len = 3
        ''' Step length for tree extention procedure'''  #Step length for extension procedure
        #self.path_nodes=[]
        #self.path_segments=[]
        #self.current_nodes=0
        

        # Map specific:

        self.mapBoundaryX = 60  # Map x length
        self.mapBoundaryY = 60  # Map y length

        self.rho_node = 5.0  #Distance from the obstacle 
        self.d_star=1  #Distance minimum from obstacle, Must be low 
        self.lamda=0.5
        # Attraction Gains:
        self.KP = 2  # Position Gain

        #Plotting functions to create a plotili plausible plot





    
    def compute_node_attractive_force(self,point:Node):
        ''' A node centric implementation of Attractive forces
         \n input: point: Node
         \n returns attractive force: (magnitude, vector:(cos(theta) ,sin(theta)) '''
        #dist = math.sqrt((point.x - self.end_node.x) * (point.x - self.end_node.y) + (point.y - self.end_node.y) * (point.y - self.end_node.y))
        dist2=((point.x - self.end_node.x)**2 + (point.x - self.end_node.y)**2)**0.5
        theta = math.atan2((self.end_node.y - point.y),(self.end_node.x - point.x))
        vector=(math.cos(theta),math.sin(theta))
        dx=1.5*vector[0]
        dy=1.5*vector[1]
        attractive_force=-2.0 * self.KP * dist2
        #print("Attractive forces generated \n")

        #return dist2,theta,vector
        return attractive_force,dx,dy
    
    
    def nearest_vertex(self,point2:Node):
        '''Returns the nearest obstacle vertex to a point. Using kd tree to store and return index of vertex and its dist
        \n input: point2:Node
        \n returns: (dist, index)'''

        p=[point2.x,point2.y]
        dist,index=self.tree.query(p)
        #vertex=self.v_p[index] #The value being returned 
        return(dist,index)


    def Sample_free(self):
        '''Sample from the configuration space'''
        delta=self.delta
        goal_sample_rate = self.goal_sample_rate

        if np.random.random() >goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                             np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
        
        return self.end_node


    def RGD(self):
        '''Randomized Gradient Descent(sort of) Algorithm taken from P-RRT by Ahq
        \n input :random samples
        \n output: modified random samples'''
        x_rand=self.Sample_free()
        attractive_forces,dx,dy=self.compute_node_attractive_force(x_rand)
        dmin=self.nearest_vertex(x_rand)
        if dmin[0]<self.d_star:
            #print("dmin accessed")
            return x_rand
        else:
            #rand_point=[]
            #print("Rand point acessed")
            x_prand=Node([x_rand.x+dx,x_rand.y+dy])
            

            return x_prand

    @staticmethod
    def nearest_neighbour(node_list,n):
        '''Find the nearest neighbour in vertex list
        \n Input: vertex_list, node
        \n Output: nodelist[nearest val]'''
        return node_list[int(np.argmin([math.hypot(nd.x-n.x,nd.y-n.y) for nd in node_list]))]


    def extend(self,start_position,end_position):
        dist=distance_calc(start_position, end_position)
        dist=min(dist,self.step_len)
        angle=math.atan2(end_position.y-start_position.y,end_position.x-start_position.x)
        
        node_new=Node(start_position.x+dist*math.cos(angle),start_position.y+dist*math.sin(angle))
        node_new.parent = start_position
        return node_new

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.end_node.x, self.end_node.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path





    def planning(self):
        '''Where da magic really happens'''
        #time.sleep(20)
        for i in range(self.iter_max):
            #
            p_rand=self.RGD()
            node_near = self.nearest_neighbour(self.vertex, p_rand)
            node_new = self.new_state(node_near, p_rand)

            if node_new and not self.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.end_node)
                if dist <= self.step_len and not self.is_collision(node_new, self.end_node):
                    #newstate
                    self.new_state(node_new, self.end_node)
                    return self.extract_path(node_new)
            
        return None


    def is_inside_obs(self, node):
        '''Checking if it is insided rectangle
        \n 
        \n Input: node
        \n Output: bool if inside true then bad
        '''
        delta = self.delta
      
        for (x, y, w, h) in self.obs_rectangle2:
            if 0 <= node.x - (x - delta) <= w + 2 * delta \
                    and 0 <= node.y - (y - delta) <= h + 2 * delta:
                return True
            

        #for (x, y, w, h) in self.obs_boundary:
         #   if 0 <= node.x - (x - delta) <= w + 2 * delta \
          #          and 0 <= node.y - (y - delta) <= h + 2 * delta:
           #     return True

        #return False
        


    def is_collision(self, start, end):
            '''Collision checker
            \n  
            '''
            if self.is_inside_obs(start) or self.is_inside_obs(end):
                return True

            o, d = self.get_ray(start, end)
            #obs_vertex = self.get_obs_vertex()

            for (v1, v2, v3, v4) in self.obs_vertix:
                if self.is_intersect_rec(start, end, o, d, v1, v2):
                    return True
                if self.is_intersect_rec(start, end, o, d, v2, v3):
                    return True
                if self.is_intersect_rec(start, end, o, d, v3, v4):
                     return True
                if self.is_intersect_rec(start, end, o, d, v4, v1):
                    return True

            return False

    def is_intersect_rec(self, start, end, o, d, a, b):
        '''If path intersects the rectangle obstacle
        \n input: start(node), end(node), vertex of rect
        '''
        v1 = [o[0] - a[0], o[1] - a[1]]
        v2 = [b[0] - a[0], b[1] - a[1]]
        v3 = [-d[1], d[0]]

        div = np.dot(v2, v3)

        if div == 0:
            return False

        t1 = np.linalg.norm(np.cross(v2, v1)) / div
        t2 = np.dot(v1, v3) / div

        if t1 >= 0 and 0 <= t2 <= 1:
            shot = Node((o[0] + t1 * d[0], o[1] + t1 * d[1]))
            dist_obs = self.get_dist(start, shot)
            dist_seg = self.get_dist(start, end)
            if dist_obs <= dist_seg:
                return True

        return False
    




    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    @staticmethod
    def get_ray(start, end):
        '''Get direction of vector b/w two nodes
        \n returns: orig(original node points), 
        \n direction( difference between them)'''
        orig = [start.x, start.y]
        direc = [end.x - start.x, end.y - start.y]
        return orig, direc

    @staticmethod
    def get_dist(start, end):
        '''Gets distance between two nodes'''
        return math.hypot(end.x - start.x, end.y - start.y)


    def animation(self, nodelist, path, name, animation=False):
        self.plot_grid(name)
        #time.sleep(5.5)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)




    #And now for functions to make plots
    def plot_grid(self, name):
        '''Initializes the plot
        \n Adds the environment features
        \n And plots the end and start points
        \n plots title, names
        \n Input: title name'''
        fig, ax = plt.subplots()
       

        for (ox, oy, w, h) in self.obs_rectangle2:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                ))
            
        for (ox, oy, w, h) in self.obs_boundary:
            ax.add_patch(patches.Rectangle((ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True))

        plt.plot(self.start_pos[0], self.start_pos[1], "bs", linewidth=3)
        plt.plot(self.end_pos[0], self.end_pos[1], "gs", linewidth=3)
        #plt.ylim(0, self.y_range[1])
        #plt.xlim(0, self.x_range[1])
        plt.title(name)
        plt.axis("equal")


    @staticmethod
    def plot_visited(nodelist, animation):
        '''Input: vertex list, animation:just a name'''
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        '''Plot the final path'''
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(1)
        plt.show()



#wait = input("Press Enter to continue.")
#time.sleep(15)
start_time = time.time()

adc=AFP_final()
point33=Node([50,50])
#print("THe nearest vertix is :  \n \n \n \n",adc.nearest_vertex(point33),"\n \n \n")
 #computing attractiev forces
#print("The attractive forces at point 20,20 are : \n \n \n \n \n",adc.compute_node_attractive_force(point33),"\n \n \n \n")  
k=adc.Sample_free()  
l=adc.RGD()  
m=adc.planning()
print("--- %s seconds ---" % (time.time() - start_time))

if m:
    #time.sleep(20)
    adc.animation(adc.vertex,m,'P-RRT by Salman',True)
    #an=animation.FuncAnimation(anim)

    #plt.show()
    #FFwriter = animation.FFMpegWriter()
    #an.save('animationP_RRT.mp4', writer = FFwriter, fps=10)

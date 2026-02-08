import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import random
import argparse
import robots
import time #b)

class Node:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

class Tree:
    def __init__(self, node_list, kdtree):
        self.node_list = node_list
        self.kdtree = kdtree
        self.size = len(node_list) #b)


    def add(self,new_point):
        
        self.node_list.append(new_point)

        self.size += 1 #b)
        
        if len(self.node_list) % 50 == 0:
            data = [n.point for n in self.node_list]
            self.kdtree = cKDTree(data)
    


class RRT:
    def __init__(self, step_size=0.1,max_iter=6000,env_name="Hardest"):
        self.controller = robots.Gen3LiteArmController(env_name=env_name)

        self.start = Node(self.controller.getCurrentJointAngles())
        self.goal = Node(self.controller.goal_angles)
        self.rand_ranges = self.controller.getRanges()
        self.step_size = step_size
        self.max_iter = max_iter
        self.start_tree = Tree([self.start],cKDTree([self.start.point]))
        self.goal_tree = Tree([self.goal],cKDTree([self.goal.point]))
        self.path_to_goal = []

        self.nodes_created = 0 #b)


    # This a "stub" planning function. It adds random nodes always
    # to the root of a start and goal tree. It shows many of the 
    # syntax elements and helper functions you could use, but 
    # you will have to fix and extend this to make it an RRT or 
    # RRT-Connect planner that can solve the harder environments.
    def plan(self):
        for k in tqdm(range(self.max_iter)):
            #alternate between start and goal trees
            if k % 2 == 0:
                tree_a, tree_b = self.start_tree, self.goal_tree
                swap = False
            else:
                tree_a, tree_b = self.goal_tree, self.start_tree
                swap = True
            
            #sample random
            x_rand = self.sample()
            
            #find nearest node in tree_a
            x_near_a = self.nearest_node(x_rand, tree_a)
            
            #steer x_near_a 
            x_new_a = self.steer(x_near_a, x_rand)
            
            #check if new_a is collision free
            if self.collision_free(x_near_a.point, x_new_a.point):

                #add new node to tree_a
                x_new_a.parent = x_near_a
                tree_a.add(x_new_a)
                self.nodes_created += 1 #b)
            
                x_near_b = self.nearest_node(x_new_a.point, tree_b)
                
                while True:
                    x_new_b = self.steer(x_near_b, x_new_a.point)
                    
                    #check if we can add this edge to tree_b
                    if self.collision_free(x_near_b.point, x_new_b.point):
                        x_new_b.parent = x_near_b
                        tree_b.add(x_new_b)
                        self.nodes_created += 1 #b)

                        #check if trees are connected within reachable step size
                        distance = np.linalg.norm(x_new_b.point - x_new_a.point)
                        if distance < self.step_size:
                            #found full path
                            if swap:
                                self.path_to_goal = self.extract_path(x_new_b, x_new_a)
                            else:
                                self.path_to_goal = self.extract_path(x_new_a, x_new_b)

                            print(f"\n\nnumber of iterations: {k+1} iterations\n\n") #b)
                            return True
                        
                        x_near_b = x_new_b
                    else:
                        #can't connect further because of a collision so we break
                        break
            
            #update the trees if we swapped
            if swap:
                self.start_tree, self.goal_tree = tree_b, tree_a
            else:
                self.start_tree, self.goal_tree = tree_a, tree_b

        #solution not found
        return False

    def sample(self):
        point = []
        for i in range(0,len(self.rand_ranges[0])):
            point.append(random.uniform(self.rand_ranges[0][i], self.rand_ranges[1][i]))
        return np.array(point)

    def nearest_node(self, point, tree):
        t0 = time.time()
        _, idx = tree.kdtree.query(point)
        dt = time.time() - t0

        return tree.node_list[idx]

    def steer(self, from_node, to_point):
        direction = to_point - from_node.point
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            new_point = to_point
        else:
            direction = direction / distance
            new_point = from_node.point + self.step_size * direction

        new_node = Node(new_point)
        new_node.parent = from_node
        return new_node

    def collision_free(self, p1, p2):
        return self.controller.collision_free(p1,p2)

    def reached_goal(self, node,goal=None):
        if goal is None:
            goal = self.goal
        
        return np.linalg.norm(node.point - goal.point) < self.step_size and self.collision_free(node.point,goal.point)

    def extract_path(self, start_node,goal_node):
        # Build the start-tree path, leaf to root (backwards) 
        start_tree_path = []
        while start_node is not None:
            start_tree_path.append(start_node.point)
            start_node = start_node.parent
        
        # Build the goal-tree path, leaf to root (forwards)
        goal_tree_path = []
        while goal_node is not None:
            goal_tree_path.append(goal_node.point)
            goal_node = goal_node.parent 

        # First add the start path, reversing
        overall_path = start_tree_path[::-1]
        # Add the goal path
        overall_path.extend(goal_tree_path)
        return overall_path

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
                    prog='arm_rrt',
                    description='Plans and executes paths for arms around obstacles.')
    parser.add_argument('--filename',default='rrt_path.npy')           
    parser.add_argument('-e', '--environment',default='Easiest')
    parser.add_argument('-p', '--plan',action='store_true',default=True)
    parser.add_argument('-r', '--run',action='store_true',default=False)
    args = parser.parse_args()

    rrt = RRT(env_name=args.environment)

    if args.plan:
        start_time = time.time() #b)
        success = rrt.plan()
        elapsed = time.time() - start_time #b)

        print(f"time elapsed: {elapsed:.3f} s") #b)
        print(f"nodes created: {rrt.nodes_created}") #b)

        if success:
            print("Tree planning reached the goal.")
            np.save(args.filename,rrt.path_to_goal)
        else:
            print("Failed to find a path to the goal.")

        rrt.controller.visTreesAndPaths( 
            [rrt.start_tree,rrt.goal_tree],             
            [rrt.path_to_goal],
            rgbas_in=[[0.5,0.0,0.5,1.0],[0.902,0.106,0.714,1.0]]
        )

    if args.run:
        path_to_goal = np.load(args.filename)
        rrt.controller.execPath(path_to_goal)
    
    
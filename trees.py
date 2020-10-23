import numpy as np
import pandas as pd
import find_split as fs


class binarySearchTree:
    def __init__(self, data, depth=0, label = None):
        self.left_child = None
        self.right_child = None
        self.depth = depth+1
        self.label = label
        self.split_value = None
        self.split_router = None
        
        if len(np.unique(data[:,-1]))==1: #assuming last column is for labels
            self.label = data[0,-1]
        
        else:
            #not all samples have same label, do a split
            # split router: router from [1,...,6]
            # split_value: router value
            split_router,split_value,temp_data=fs.find_split(data)
            self.split_value=split_value
            self.split_router=split_router
            l_data=temp_data[0]
            r_data=temp_data[1]
             
            #recursively search the tree, branching into 2 
            self.left_child=binarySearchTree(l_data, self.depth)
            self.right_child=binarySearchTree(r_data, self.depth)
        
    def get_max_depth(self):
        #search each branch recursively and get max depth
            max_depth=[self.depth]
            if self.left_child:
                max_depth.append(self.left_child.get_max_depth())
            if self.right_child:
                max_depth.append(self.right_child.get_max_depth())                 
            return max(max_depth)
        
    def predict_one(self,data):
        if self.label:
            return np.array([int(self.label)])
        else:
            if data[self.split_router-1]<=self.split_value:
                return self.left_child.predict_one(data)

            else:
                return self.right_child.predict_one(data)            
            
    def predict(self,data):
        data=np.squeeze(data)
        
        if len(data.shape)>1:
            return np.array([self.predict_one(i) for i in data]).flatten()
        
        else:
            return self.predict_one(data).flatten()

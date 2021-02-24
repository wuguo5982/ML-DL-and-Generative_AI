from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}       

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        self.X =X
        self.y =y  
        self.tree = best_split(self.X,self.y)
        split(self.tree)
        return self.tree

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label    
        pred = newPredict(self.tree, record)
        return pred

def best_split (X,y):
    max_gain = 0          #  To find the best information gain
    new_val=0                #  val is split_val
    new_col=0                # col is split_attribute
    newX_current =[]
    newy_current =[]    
    ncol = len(X[0])      # number of columns
    values = []           # list of mean of split_val
    Node = {}
    for col in range(ncol):        # Calculate the mean for split_val.
        item = np.mean([row[col] for row in X if type(row[col] is not str)])
        if not np.isnan(item):
            values.append(item)     
        
        for val in values:
             # col is split_attribute, val:split_val                  
            X_left, X_right, y_left, y_right = partition_classes(X, y, col, val) 
            if len(X_left) == 0 or len(X_right) == 0:
                continue

            # Update 
            X_current = [X_left, X_right]
            y_current = [y_left, y_right]
            gain = information_gain(y,  y_current)            

            if gain >= max_gain:
                max_gain = gain
                new_val = val
                new_col = col
                newX_current = X_current
                newy_current = y_current
                Node['ind'] = new_col
                Node['value'] = new_val
                Node['X_child'] = newX_current
                Node['y_child'] = newy_current
    return Node
          

def split(Node):
    X_left, X_right = Node['X_child']
    y_left, y_right = Node['y_child']

    if not X_left or not X_right:
        Node['left'] = Node['right'] = (y_left + y_right)[0]
    
    # For left child:
    if entropy(y_left) <0.5:    
        Node['left'] = np.argmax(np.bincount(y_left))
    else:
        Node['left'] = best_split(X_left, y_left)
        split(Node['left'])
        
    # For right child:
    if entropy(y_right) <0.5: 
        Node['right'] = np.argmax(np.bincount(y_right))
    else:
        Node['right'] = best_split(X_right, y_right)
        split(Node['right'])
    return
                

def newPredict(Node, Rec):
    if Rec[Node['ind']] <= Node['value']:
        if Node['left'] is dict:
            return newPredict(Node['left'], Rec)
        else:
            return Node['left']
    else:
        if Node['right'] is dict:
            return newPredict(Node['right'], Rec)
        else:
            return Node['right']
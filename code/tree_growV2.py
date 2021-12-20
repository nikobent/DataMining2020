# Ioannis Tsogias 6966985
# Nikolaos Bentis 6662889
# Markos Polos 6943721
import random
from random import randint

import numpy as np
from anytree import Node, RenderTree
from numpy import genfromtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as conf
from anytree.dotexport import RenderTreeGraph

from anytree.exporter import DotExporter


def tree_grow(x, y, nmin=8, minleaf=3, nfeat=None):
    """Given the X and Y arrays and the parameters of the splitting. This algorithm tries to find for each Node that it
    examines the best possible way to split  the instances (both on column and value). First it doesn't have any node
    and creates (if possible) the root node, after it splits the node into to children Nodes the nodes are inputted into
    a lifo queue and wait for their turn to be examined. For any node the process is the same as the root node. They
    either get split into two children nodes or if its not possible due to the parameters (minleaf, nmin) the node
    becomes a leaf node. When this queue is empty and there is no node left to examine, the algorithm stops and the tree
    is returned.

    Args:
        x (ndarray): 2D array containing the attribute variables
        y (ndarray): 1D array containing the class values of each row
        nmin (int): Min observations for a node to split
        minleaf (int): Min observations for a leaf node
        nfeat (int): Number of features to be considered for each split

    Returns:
        Node: The classification tree
    """
    # If nfeat is None, we should use the total number of parameters - which is the number of columns of x
    if nfeat is None:
        nfeat = x.shape[1]
    parent = None  # For the first iteration of the while loop, so it creates the root node
    nodes_to_examine = []  # A list that will contain every node that should be examined to either split or become leaf
    x_to_examine = []
    y_to_examine = []
    node_counter = 2  # For id between nodes
    # The while-loop will stay True until the variable nodes_to_examine becomes empty and there is no other node to be
    # examine
    while True:
        # Only one time the if will hold - so we initiate the tree with the root node
        if parent is None:
            # Calling the best_split function to get for the original data the point of the best split (column, value)
            target_col, target_value = best_split(x, y, nmin, minleaf, nfeat)
            root = Node('root', rule="x%d =< %.3f" % (target_col, target_value), split_value=target_value,
                        split_column=target_col)  # Create the root node
            # Split the X and Y arrays into their left and right splits as instructed by the best split
            x_l = x[x[:, target_col] > target_value, :]
            x_r = x[x[:, target_col] <= target_value, :]
            y_l = []
            y_r = []
            helper = x[:, target_col]
            for i in range(len(helper)):
                if helper[i] > target_value:
                    y_l.append(y[i])
                else:
                    y_r.append(y[i])
            # For each new node created we calculate the number of instances for each class inside it.
            # When a node is a leaf we use these values to estimate the majority class
            number_of_class_a_split_1 = sum(y_l)
            number_of_class_b_split_1 = len(y_l) - number_of_class_a_split_1
            number_of_class_a_split_2 = sum(y_r)
            number_of_class_b_split_2 = len(y_r) - number_of_class_a_split_2
            # Create the child nodes of the root. Each node has an id and a linkage to the parent.
            # Split_value and column are the values are none if the node becomes a leaf else are the splitting points.
            # Its own x and y after it created and the instances for each class inside it
            child_left = Node('c1', parent=root, rule=None, split_value=None, split_column=None,
                              value=[number_of_class_a_split_1, number_of_class_b_split_1])
            child_right = Node('c2', parent=root, rule=None, split_value=None, split_column=None,
                               value=[number_of_class_a_split_2, number_of_class_b_split_2])
            root.children = [child_left, child_right]  # link between root and child nodes
            # Both appended into the list of nodes to be examined
            nodes_to_examine.append(child_left)
            nodes_to_examine.append(child_right)
            x_to_examine.append(x_l)
            x_to_examine.append(x_r)
            y_to_examine.append(y_l)
            y_to_examine.append(y_r)
            parent = 1  # Root initiated and now parent is not None
        # The loop for any other node out of the root
        else:
            # Check if list of nodes to be ex. is empty, if it is break the loop as there are no other nodes to examine
            if not nodes_to_examine:
                break
            # Get the first node from the list and then deleted from the list
            # Usage as a lifo queue
            candidate = nodes_to_examine[0]
            nodes_to_examine = nodes_to_examine[1:]
            # Get the X and Y instances that belong to the node so we can examine if it can split and where
            x_c = x_to_examine[0]
            y_c = y_to_examine[0]
            x_to_examine = x_to_examine[1:]
            y_to_examine = y_to_examine[1:]
            # Second check
            target_col, target_value = best_split(x_c, y_c, nmin, minleaf,
                                                  nfeat)  # Call of best split to examine the node
            # If None then the examined Node becomes a leaf and continue the while loop with another node if available
            if target_value is None and target_col is None:
                continue
            # Else split the node accordingly
            else:
                node_counter = node_counter + 1  # For the ids to progress properly
                # Split the X and Y arrays into their left and right splits as instructed by the best split
                x_l = x_c[x_c[:, target_col] > target_value, :]
                x_r = x_c[x_c[:, target_col] <= target_value, :]
                y_l = []
                y_r = []
                helper = x_c[:, target_col]
                for i in range(len(helper)):
                    if helper[i] > target_value:
                        y_l.append(y_c[i])
                    else:
                        y_r.append(y_c[i])
                # For each new node created we calculate the number of instances for each class inside it
                # When a node is a leaf we use these values to estimate the majority class
                number_of_class_a_split_1 = sum(y_l)
                number_of_class_b_split_1 = len(y_l) - number_of_class_a_split_1
                number_of_class_a_split_2 = sum(y_r)
                number_of_class_b_split_2 = len(y_r) - number_of_class_a_split_2
                # Creation of the two child nodes
                child_left = Node('c%d' % node_counter, parent=candidate, split_value=None,
                                  split_column=None,
                                  value=[number_of_class_a_split_1, number_of_class_b_split_1])
                node_counter = node_counter + 1  # For ids to progress properly
                child_right = Node('c%d' % node_counter, parent=candidate, split_value=None,
                                   split_column=None,
                                   value=[number_of_class_a_split_2, number_of_class_b_split_2])
                # Link between children and parent nodes
                candidate.children = [child_left, child_right]
                # Update the split_column and split_value which are None when a node is initiated
                # If their values remain None that means that the node is a leaf
                candidate.split_column = target_col
                candidate.split_value = target_value
                candidate.rule = "x%d =< %.3f" % (target_col, target_value)
                # Input the children nodes into the list
                nodes_to_examine.append(child_left)
                nodes_to_examine.append(child_right)
                x_to_examine.append(x_l)
                x_to_examine.append(x_r)
                y_to_examine.append(y_l)
                y_to_examine.append(y_r)
    return root


def tree_pred(x, tree):
    """Given instances X and a tree. For each of the instances, the function starts from the root and following the
    comparisons happening on the nodes, it arrives on a leaf. There it finds the majority class and for that instance
    this is the predicted class. After all the instances have a prediction the function returns the list.

    Args:
        x (ndarray): 2D array containing the instances to have their class predicted
        tree (Node): The root of the tree

    Returns:
        list: The list with the predicted classes
    """
    predictions = []  # Empty list to store the predictions
    for xpred in x:  # For each instance
        p = tree  # Start with the root node
        while True:  # Until a leaf is found the while wil continue
            # If p.split_column is None that means that the Node is a leaf and we estimate the majority class
            if p.split_column is None:
                # After finding majority class we break the while loop
                if p.value[0] > p.value[1]:
                    predictions.append(1)
                    break
                else:
                    predictions.append(0)
                    break
            # Else by taking the splitting values of the node move to its left or right children node
            else:
                # The column the split happens and its value
                col = p.split_column
                val = p.split_value
                if xpred[col] > val:
                    new_p = p.children[0]
                else:
                    new_p = p.children[1]
                # Find the new node and continue the while loop with that
                p = new_p
    return predictions


def tree_grow_b(x, y, m, nmin=8, minleaf=3, nfeat=None):
    """The function will produce a list of m trees. On each iteration the function creates new X and Y, which have the
    same size as the original input but has random samples from X and Y in it (many instances are duplicates). Then with
    these it creates a tree by calling the tree_grow algorithm with the other parameters. After m trees are created and
    stored into a list, it returns the list.

    Args:
        x (ndarray): 2D array containing the attribute variables
        y (ndarray): 1D array containing the class values of each row
        m (int): Number of trees
        nmin (int): Min observations for a node to split
        minleaf (int): Min observations for a leaf node
        nfeat (int): Number of features to be considered for each split

    Returns:
        List of trees (their root Nodes)
    """
    trees = []  # The list to store the trees
    dim = x.shape[0]  # The number of instances in X, so the new Xs have the same number
    for i in range(m):  # m times -> m trees
        # The new X and Y for this iteration
        x_local = []
        y_local = []
        # For j = 0 until the number of instances of X
        for j in range(dim):
            # Each time a random instance is selected and stored into the new X and Y
            ind = randint(0, dim - 1)
            x_local.append(x[ind, :])
            y_local.append(y[ind])
        # Call tree_grow and append the tree into the list
        tree = tree_grow(np.array(x_local), y_local, nmin, minleaf, nfeat)
        trees.append(tree)
    return trees


def tree_pred_b(x, trees):
    """Given instances X and a list of trees. For each of the instances, the function calls the tree_grow function for
    each tree and sums up the results. The predicted class of the instances is the one that most trees predict.
    After all the instances have a prediction the function returns the list.

    Args:
        x (ndarray): 2D array containing the instances to have their class predicted
        trees (list): List of trees

    Returns:
        list: The list with the predicted classes
    """
    predictions_bag = []  # List that will store the voted predictions
    predictions_gathered = []  # List that will hold a list of predictions for each tree
    m = len(trees)
    # For all instances of X we estimate the predictions based on each tree by calling tree_pred -> list of results for
    # each tree store these lists into the predictions_gathered, which contains for each tree -> the predictions for all
    # X
    for i in range(m):
        predictions = tree_pred(x, trees[i])
        predictions_gathered.append(predictions)
    # For each instance of X we sum the predictions made from all trees (either 1 or 0)
    # If the sum divided by the total number of trees is above 0.5 the class is 1 else is 0
    for j in range(x.shape[0]):
        summed = 0
        for k in range(m):
            summed = summed + predictions_gathered[k][j]
        if summed / m > 0.5:
            predictions_bag.append(1)
        else:
            predictions_bag.append(0)
    return predictions_bag


def best_split(x, y, nmin, minleaf, nfeat):
    """Given X and Y, we try to find the point (column and value) of the best possible way to split. With best possible
    we mean the point where we get the best quality, whichs means the biggest reduction of impurity from the parent node
    to the children nodes. So we examine each column and find the best split for it -- must respect minleaf, nmin and is
    created by nfeat params. After all columns are examined we have kept the best quality, its column and splitpoint and
    we return the last two.

    Args:
        x (ndarray): 2D array containing the attribute variables
        y (ndarray): 1D array containing the class values of each row
        nmin (int): Min observations for a node to split
        minleaf (int): Min observations for a leaf node
        nfeat (int): Number of features to be considered for each split

    Returns:
        int: The number of the column that the split happened
        float: The value of comparison for the split
    """
    parent_impurity = gini_index(y)  # Call the gini index function to get the impurity of the parent
    number_of_col = x.shape[1]  # Possible columns for split
    # If target and col_target remain None then the Node is a leaf
    target = None
    col_target = None
    # To find the best split point
    best_quality = 0
    # If parent has lower instances than nmin this node becomes a leaf -- return 2 Nones (Overfitting param)
    if x.shape[0] < nmin:
        return target, col_target
    # Check nfeat parameter. Either all features will be examined or a random sample of them with length of nfeat
    # Either way feat is a list of integers which are the columns(features) to be examined
    if nfeat == number_of_col:
        feat = list(range(number_of_col))  # list from 0 to nfeat-1
    else:
        feat = random.sample(range(0, number_of_col - 1),
                             nfeat)  # list with nfeat random integers from 0 to number_of_col -1
    # Examine each feature
    for j in feat:
        xcolumn = x[:, j]  # X values only for the examined column
        # First we sort them and the points that will be examined for a split are the midpoints between two consecutive
        # values
        x_sorted = np.sort(np.unique(xcolumn))
        x_splitpoints = []
        # Create the list of the midpoints that will be examined as possible target values
        for i in range(x_sorted.shape[0] - 1):
            x_splitpoints.append((x_sorted[i] + x_sorted[i + 1]) / 2)
        qualities = np.array([])
        x_kept = []
        # For each of the possible split-points we estimate the impurity of the child nodes and calculate the quality of
        # the split
        for point in x_splitpoints:  # Examine all possible split-points
            child1 = []
            child2 = []
            # Separate the Y instances to the child nodes according to the rule
            for i in range(len(xcolumn)):
                if xcolumn[i] > point:
                    child1.append(y[i])
                else:
                    child2.append(y[i])
            # Check if the split is possible according to minleaf constrains -- Overfitting param
            # If one of the children has a number of instances lower than minleaf examine other split-points
            if len(child1) < minleaf or len(child2) < minleaf:
                continue
            ratio = len(child1) / len(y)
            # Store all the calculated qualities into an array and keep the split-points that that took place
            qualities = np.append(qualities,
                                  parent_impurity - ratio * gini_index(child1) - (1 - ratio) * gini_index(child2))
            x_kept.append(point)
        # If empty this column can't offer a split
        if not list(qualities):
            continue
        # Find the max quality and its split-point for this column
        candidate = np.max(qualities)
        ind = np.argmax(qualities)
        # Compare if the candidate quality is better than the stored one
        if candidate > best_quality:
            best_quality = candidate
            target = x_kept[ind]
            col_target = j
    # Return the column and its split-point
    return col_target, target


def gini_index(labels):
    """Estimate for the list of labels how impure it is. This formula works only for binary classes. If both classes
    have the same number --> 0.5, else lower.

    Args:
        labels (list): List of the class of each instance

    Returns:
        float: Impurity score
    """
    labels = list(labels)
    numerator = np.sum(labels)
    div = len(labels)
    return (numerator / div) * (1 - numerator / div)


# Print a decision tree
def print_tree(t):
    print(RenderTree(t))
    # DotExporter(t).to_dotfile('udo.dot')
    # Source.from_file('udo.dot')
    # render('dot', 'png', 'udo.dot')


dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]

credit_data = [
    [22, 0, 0, 28, 1, 0],
    [46, 0, 1, 32, 0, 0],
    [24, 1, 1, 24, 1, 0],
    [25, 0, 0, 27, 1, 0],
    [29, 1, 1, 32, 0, 0],
    [45, 1, 1, 30, 0, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 1, 1],
    [63, 1, 1, 58, 0, 0],
    [36, 1, 0, 52, 1, 1],
    [23, 0, 1, 40, 0, 1],
    [50, 1, 1, 28, 0, 1]
]

# dataset = np.array(credit_data)

pima = genfromtxt('pima_numbers.csv', delimiter=',')

x = pima[:, :-1]
y = pima[:, -1]
tree = tree_grow(x, y, 20, 5)
print_tree(tree)

pred = tree_pred(x, tree)
print(conf(y, pred))
print(accuracy_score(y, pred))

Ben Orkild
u1196243
CS6350, Fall 2023

This code is for HW1 in CS6350. The runHW1.sh file runs the python code in DT_Practice.py 

DT_Practice.py has all the python code necessary for the assignment, including an implementation 
of a tree class and the ID3 algorithm for building decision trees

Learning Decision Trees:

This toolkit implements the ID3 algorithm to learn decision trees.

To learn a tree, call:

tree = ID3(data, Attributes, AttributeVals, AttIdx, max_depth, gainFunction)

Where:
data - data to train tree, with the attributes in the columns and labels in the rightmost column NOTE: data should all be strings
Attributes - list of strings of possible attributes
AttributeVals - list of lists that contains the possible values for each attribute
AttIdx - the column index in data that each entry of Attributes corresponds to
max_depth - Optional: The maximum depth the tree will grow to (default is to grow until stop conditions are met)
gainFunction- Optional: Specific the gain function to use (default is information gain)

Other gain functions: 
Gini Index: gainFunction=GI
Majority Error: gainFunction=ME


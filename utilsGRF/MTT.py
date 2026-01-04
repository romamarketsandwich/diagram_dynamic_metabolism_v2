
#!/usr/bin/env python

"""
Copyright (C) 2012, 2013 Tobias Ahsendorf, Felix Wong, Roland Eils, Jeremy Gunawardena
Contact <felixjwong@gmail.com> or <jeremy@hms.harvard.edu>.
Released under the GNU General Public License Version 3

This file is part of a code package (hereafter referred to as The Code) 
for the paper "A framework for modeling gene regulation that accommodates 
nonequilibrium mechanisms", by the same authors, submitted, 2013.

The Code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with The Code.  If not, see <http://www.gnu.org/licenses/>.
"""

# MTT.py - for python 2.7+
# Version 1.0

# Algorithm from Takeaki Uno, "An algorithm for enumerating all directed
# spanning trees in a directed graph", Springer LNCS 1178, 1996.

# Usage: 'python MTT.py [input graph txt file]' at the command line
# E.g. 'python MTT.py graph.txt'
# graph.txt should be a text file containing a description of the graph 
# as a list of triples (i,a,j), denoting an edge from vertex i to vertex
# j with label a. 

# Output: for each vertex i, the spanning trees rooted at i are listed in 
# the file graph-i.txt. A cleaned version of the graph description is also
# written to graph-parsed.txt for easier processing by the accompanying 
# Mathematica notebook.

# For further details see the website at vcp.med.harvard.edu/software.html

# 
#-----------------------------------------------------------------------------#

# packages
from __future__ import print_function
import sys
import pickle
import re
global labels, numberofnodes, edges
if len(sys.argv) < 2:
    sys.exit("Usage: 'python MTT.py man' or 'python MTT.py [input graph txt file]'")

# manual for this script
if sys.argv[1] == 'man':
    sys.exit("\nUsage: python MTT.py [input graph txt file]\nThis script runs Uno's algorithm on any strongly connected digraph to output a list of all monomials in each rho_i, separated into <graph file>-i.txt. The correspondence is given in <graph file>-labels.txt.\n\nNOTE: The graph in the input file should be specified in the following format:\n(a1,k1,b1)[whitespace or comma](a2,k2,b2)[whitespace or comma](a3,k3,b3) etc.\nAll lines that do not begin with the character '(' are not parsed. The numbering of a1 should begin with 1.\n")

# input the graph from file specified in command line and eliminate spaces
inputfile = sys.argv[1]
fopen = open(inputfile);
x = []
z = []
rawedge_str = ''
for line in fopen.readlines():
    y = [''.join(value.split()) for value in line.split('\n')]
    z = z + y
fopen.close()
for item in z:
    if len(item) > 0:
        #        print item
        if item[0] == '(':
            rawedge_str = rawedge_str + item
rawedge_str = rawedge_str.replace(')(','),(').replace("'",'').replace('"','')

# parse the string of edges to add quotes as needed
alledges = ''
commaplace = 0
for i in range(0,len(rawedge_str)):
    addstr = rawedge_str[i]
    if rawedge_str[i] == ',':
        commaplace = commaplace + 1
        if commaplace % 3 == 1:
            addstr = addstr + "'"
        if commaplace % 3 == 2:
            addstr = "'" + addstr
    alledges = alledges + addstr

x.append(alledges)

try:
    r = eval(x[0])
except SyntaxError:
    sys.exit('\nError: the edges of the graph did not parse correctly. Please double check that you have inputted the following correctly:\n'+x[0]+'\nThis line should be in the format (a1,"k1",b1),...,(an,"kn",bn).\nCall "python '+sys.argv[0]+' man" for detailed usage instructions.\n')

# assuming that the string of edges is parsed correctly, we define the following:
# make edges
alllabels = []
edgelist = []
for item in r:
    edgelist.append((item[0],item[2]))
    alllabels.append(item[1])
edges = set(edgelist)

# auxiliary function for getting unique labels in order
def getunique(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

uniquelabels = getunique(alllabels)
labels = [0] * (len(uniquelabels))
for i in range(0,len(uniquelabels)):
    addtolabel = []
    for item in r:
        if item[1] == uniquelabels[i]:
            addtolabel.append((item[0],item[2]))
    labels[i] = set(addtolabel)
numberofnodes = max(max(edges))

# also initialize the rho's
global rhos
rhos = [0]*(numberofnodes+1)
for i in range(1,numberofnodes+1):
    rhos[i] = []

#-----------------------------------------------------------------------------#

# Declare a few functions for implementing the algorithm:

# depth first search function
def dfs(root):
    global numberofnodes, edges, explored, discover, backedge
    global transversededges, vertexorder
    explored = explored + [root]
    
    # adjacent edges have the root as their heads
    adjacentedges = ({item for item in edges if item[1]==root})
    adjacentedgessorted = []
    
    # organize adjacent edges in order of their tails
    for i in range(0,numberofnodes+1):
        for item in adjacentedges:
            if item[0]==i:
                adjacentedgessorted = adjacentedgessorted + [item]
    
    # label these edges into discovery or backedges, as in usual
    # implementations of dfs
    for item in adjacentedgessorted:
        if not item in transversededges:
            if not item[0] in explored:
                discover = discover + [item]
                # vertexorder keeps track of the transversal route of depth
                # first search
                vertexorder = vertexorder + [item[0]]
                dfs(item[0])
            if item[0] in explored:
                backedge = backedge + [item]
        transversededges = transversededges + [item]

#-----------------------------------------------------------------------------#

# this function classifies the edges not in the current spanning tree into the
# sets backarc and nonbackarc, as prescribed by the Uno algorithm
def classifyarcs():
    global numberofnodes, edges, explored, discover, backedge
    global transversededges, backarc, nonbackarc
    
    # check which edges have tails that are ancestors of their heads we do this
    # using 'pointer' that traverses the current spanning tree; rather
    # self-explanatory
    for item in ({x for x in edges if not x in discover}):
        pointer = item[1]
        i = 0
        while i < numberofnodes+1:
            for treeedge in discover:
                if pointer == treeedge[0]:
                    pointer = treeedge[1]
                    if pointer == item[0]:
                        backarc = backarc + [item]
            i = i+1
    
    # if not a backarc, then a nonbackarc
    for item in ({x for x in edges if not x in discover}):
        if not item in backarc:
            nonbackarc = nonbackarc + [item]

#-----------------------------------------------------------------------------#

# this function sorts backarc and nonbackarc in order of their indices, which
# is the order in which the vertices are transversed in the depth-first search
# generating the current spanning tree
def sortindices():
    global numberofnodes, backarc, nonbackarc, backarcsorted
    global nonbackarcsorted, vertexorder, discover
    for i in range(0,numberofnodes):
        for item in backarc:
            if item[0]==vertexorder[i]:
                backarcsorted = backarcsorted + [item]
        for item in nonbackarc:
            if item[0]==vertexorder[i]:
                nonbackarcsorted = nonbackarcsorted + [item]

#-----------------------------------------------------------------------------#


# given a spanning tree, this function finds the vertex it's anchored at
def findanchor(Tc):
    for i in range(1,numberofnodes+1):
        edgetovertex = ({item for item in Tc if item[0]==i})
        if edgetovertex == set([]):
            return i

#-----------------------------------------------------------------------------#

# this is a modification of the DFS function above, taking into account a
# modified edgeset instead of all the edges as in the first one. this is
# needed because when we call the first algorithm again, we are not calling it
# on our original graph G to make T0, but on Tp (but we're really just lazy)
def dfsmod(root, edgeset):
    global numberofnodes, explored, discover, backedge
    global transversededges, vertexorder
    explored = explored + [root]
    adjacentedges = ({item for item in edgeset if item[1]==root})
    adjacentedgessorted = []
    for i in range(0,numberofnodes+1):
        for item in adjacentedges:
            if item[0]==i:
                adjacentedgessorted = adjacentedgessorted + [item]
    for item in adjacentedgessorted:
        if not item in transversededges:
            if not item[0] in explored:
                discover = discover + [item]
                vertexorder = vertexorder + [item[0]]
                dfsmod(item[0],edgeset)
        transversededges = transversededges + [item]

#-----------------------------------------------------------------------------#

# this is the body of the first algorithm, referred to as
# Enum_Directed_Spanning_Trees(G) in the Uno paper
def first_algorithm(Tc):
    # initialize all our variables:
    global explored, discover, backedge, transversededges, vertexorder
    global backarc, nonbackarc, backarcsorted, nonbackarcsorted
    explored = []
    discover = []
    backedge = []
    transversededges = []
    # of course the anchor vertex is transversed first in any DFS
    vertexorder = [findanchor(Tc)]
    
    # use the DFS modification on the anchor node with the spanning tree as
    # the edgeset
    dfsmod(findanchor(Tc), Tc)
    
    # initialize sets of backarc and nonbackarc, then use classifyarcs() to
    # fill these sets up
    backarc = []
    nonbackarc = []
    classifyarcs()
    
    # intialize sorted sets of backarc and nonbackarc, then fill them up
    backarcsorted = []
    nonbackarcsorted = []
    sortindices()
    
    # now our current spanning tree is the spanning tree generated by DFS,
    # called discover
    Tp = discover
    
    # call the second algorithm, or Enum_Directed_Spanning_Trees_Iter(Tp)
    # in Uno's paper
    second_algorithm(Tp)

#-----------------------------------------------------------------------------#

# this is the body of the second algorithm, which includes
# Enum_Directed_Spanning_Trees_Iter(Tp) in Uno's paper along with a few
# necessary intialization steps
def second_algorithm(Tp):
    global T0, vertexorder, numberofnodes
    allindices = [999]
    
    # find the set T0-Tp, and calculate its minimum index v*(Tp)
    T0dTp= ({item for item in T0 if not item in Tp})
    
    # start with v*(T0)=infinity, as stipulated by Uno's paper
    if T0dTp == set([]):
        minindex = 99999
    if not T0dTp == set([]):
        for i in range(0,numberofnodes):
            for item in T0dTp:
                if vertexorder[i] == item[0]:
                    allindices = allindices + [i]
        minindex = min(allindices)
    
    # after we set minindex, we call the main sequence of the second algorithm
    Enum_Directed_Spanning_Trees_Iter(Tp, minindex)

#-----------------------------------------------------------------------------#

# this function simply finds the index of some edge given the vertexorder
def findindex(somearc):
    global vertexorder, numberofnodes
    for i in range(0,numberofnodes):
        if vertexorder[i]==somearc[0]:
            return i

#-----------------------------------------------------------------------------#

# this function counts each label in each spanning tree and outputs an array
# of counters, formatted for mathematica inputting
def makecounts(setoftrees):
    for x in setoftrees:
        reformat = []
        for item in x:
            for i in range(0,len(labels)):
                if item in labels[i]:
                    reformat.append(i)
    countingarray = [0]*len(labels)
    for i in range(0,len(labels)):
        countingarray[i] = countingarray[i] + reformat.count(i)
    stringprint=''

    for i in range(0,len(countingarray)):
        if not i == len(countingarray)-1:
            stringprint = stringprint+str(countingarray[i])+'\t'
        else:
            stringprint = stringprint+str(countingarray[i])
    return stringprint

#-----------------------------------------------------------------------------#

# this is the main content of the second algorithm, and matches the definition
# given in Uno's paper
def Enum_Directed_Spanning_Trees_Iter(Tp, minindex):
    global nonbackarcsorted, allspanningtrees, vertexorder
    
    # get set of *valid* nonbackarcs:
    newset = ({item for item in nonbackarcsorted if findindex(item)<minindex})
    
    # for each valid nonbackarc f of Tp:
    for f in newset:
        Tc = []
        Tcsorted = []
        
        # construct Tc by adding f and removing an arc with the same tail
        for item in Tp:
            if not item[0]==f[0]:
                Tc = Tc + [item]
        Tc = Tc + [f]
        for i in range(0,numberofnodes+1):
            for item in Tc:
                if item[0]==i:
                    Tcsorted = Tcsorted + [item]
        
        # if the spanning tree that we get is a new one, we output it
        if not Tcsorted in allspanningtrees:
            outprint = makecounts([Tcsorted])
            
            # add to GRFstrees if stree is relevant for numerator
            global rhos
            rhos[findanchor(Tcsorted)] = rhos[findanchor(Tcsorted)] + [outprint]
            
            global counter
            counter = counter + 1
            allspanningtrees = allspanningtrees + [Tcsorted]
            
            # finally, we call the first algorithm, or
            # Enum_Directed_Spanning_Trees(Tc) in the paper, recursively
            first_algorithm(Tcsorted)

#-----------------------------------------------------------------------------#

# main script - this executes the algorithm for our entire digraph, and
# outputs all the results

# first declare the anchorset as the set of all nodes: this is necessary because
# the algorithm only outputs given an anchor node, so we will loop over all
# anchor nodes to get all outputs for any digraph
anchorset = range(1,numberofnodes+1)
global totalnumber
totalnumber = 0

for anchor in anchorset:
    global counter
    counter = 0
    
    # define and initialize global variables needed for dfs
    global explored, discover, backedge, transversededges, backarc, nonbackarc
    global vertexorder, backarcsorted, nonbackarcsorted
    explored = []
    discover = []
    backedge = []
    transversededges = []
    backarc = []
    nonbackarc = []
    
    vertexorder = [anchor]
    backarcsorted = []
    nonbackarcsorted = []
    
    global allspanningtrees
    allspanningtrees = []
    
    # get our first spanning tree by depth first search
    dfs(anchor)
    
    # if our first spanning tree is good, we run the routine and classify
    # arcs into backarcset and nonbackarc set, and sort by indices:
    if len(discover) == numberofnodes - 1:
        classifyarcs()
        sortindices()
        
        # declare T0
        global T0
        T0 = discover
        
        # sort T0
        T0sorted = []
        for i in range(0,numberofnodes+1):
            for item in T0:
                if item[0]==i:
                    T0sorted = T0sorted + [item]
        outT0 = makecounts([T0sorted])
    
        # also for rho's
        rhos[findanchor(T0sorted)]= rhos[findanchor(T0sorted)]+[outT0]
                
        counter = 1
        
        # now use the algorithm
        first_algorithm(T0sorted)
    
    # counter for total number of spanning trees
    totalnumber = totalnumber + counter

# print individual rho's
for i in range(1,max(max(edges))+1):
    outputfle = inputfile.replace('.txt','')+'-'+str(i)+'.txt'
    logr = open(outputfle, 'w')
    for item in rhos[i]:
        print (item, file = logr)

# print parsed graph for Mathematica
outputfle = inputfile.replace('.txt','')+'-parsed.txt'
logr = open(outputfle, 'w')
print (str(x[0]).replace("'",""), file = logr)

#---------------------------end of script--------------------------------------#

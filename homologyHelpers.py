from total2partial import *
import time 
import re
import json
import numpy as np
from itertools import product
from itertools import permutations
import os
import sys

# for atomic parameter node and parameter node

def loadAtomicChainComplex(fpath):
    ret = dict()
    with open(fpath,'r') as file:
        atomicCC = json.load(file)
    for key in atomicCC:
        newk = key[1:-1].split(',')
        newk = tuple([int(x) for x in newk])
        v = atomicCC[key]
        newvalue = dict()
        for dim in v:
            newfaces = set([tuple(x) for x in v[dim]])
            newvalue[int(dim)] = newfaces
        ret[newk] = newvalue 
    return ret

def integerPartitionHelper(Sum,UpperBounds,path,ret):
    if len(path) == len(UpperBounds):
        if Sum!=0:
            return
        else:
            ret.append(path.copy())
            return 
    
    for i in range(min(Sum,UpperBounds[len(path)])+1):
        path.append(i)
        integerPartitionHelper(Sum-i,UpperBounds,path,ret)
        #print(path)
        path.pop()
        
    return 
            

def integerPartition(Sum, UpperBounds):
    ret = []
    path = []
    integerPartitionHelper(Sum,UpperBounds,path,ret)
    return ret
    

def tensorProduct(atomicParameterNodeList):
    upperBounds = [len(x)-1 for x in atomicParameterNodeList]
    topDim = sum(upperBounds)
    faceLattice = dict()
    for i in range(topDim+1):
        partition = integerPartition(i,upperBounds)
        faces = set()
        for p in partition:
            #print(p)
            facelist = [atomicParameterNodeList[j][p[j]] for j in range(len(p))]
            faces |= set(product(*facelist))
        faceLattice[i] = faces
        
    return faceLattice 

def tensorProductForDim(atomicParameterNodeList,dim):
    upperBounds = [len(x)-1 for x in atomicParameterNodeList]
    topDim = sum(upperBounds)
    faceLattice = dict()
    
    partition = integerPartition(dim,upperBounds)
    faces = set()
    for p in partition:
        #print(p)
        facelist = [atomicParameterNodeList[j][p[j]] for j in range(len(p))]
        faces |= set(product(*facelist))
    faceLattice[dim] = faces
        
    return faceLattice 




def parameterNodeLoader(fileNameList):
    """For global parameter node"""
    pnlist = []
    for fname in fileNameList:
        with open(fname,'r') as file:
            currentParameterNode = json.load(file)
            recover = dict()
            for key in currentParameterNode:
                newkey = int(key)
                newvalue = []
                for face in currentParameterNode[key]:
                    tempFace = tuple([tuple(x) for x in face])
                    newvalue.append(tempFace)
                newvalue =set(newvalue)
                recover[newkey] = newvalue
            pnlist.append(recover)
            
    return pnlist

# need the dimension to be 1
def checkFace(faceCell,bodyCell):
    for i in range(len(faceCell)):
        if not set(faceCell[i]).issubset(set(bodyCell[i])):
            return False
    return True


def unionParameterNodeFaceLattice(pnlist):
    # left multiplication sparse matrix
    ret = []
    topDim = max(pnlist[0].keys())
    
    faceCells = dict()
    bodyCells = dict()
    for i in range(len(pnlist)):
        for vertex in pnlist[i][0]:
            if vertex not in bodyCells:
                bodyCells[vertex] = len(bodyCells)
                
    for d in range(1,topDim+1):
        diffMatrix = dict()
        faceCells = bodyCells
        bodyCells = dict()
        # generate d-dimensional faces set
        for i in range(len(pnlist)):
            for dFaces in pnlist[i][d]:
                if dFaces not in bodyCells:
                    bodyCells[dFaces] = len(bodyCells)
        
        for bodyCell in bodyCells:
            for faceCell in faceCells:
                if checkFace(faceCell,bodyCell):
                    row = bodyCells[bodyCell]
                    col = faceCells[faceCell]
                    diffMatrix[(row,col)] = 1
        
        ret.append(diffMatrix)
    
    return ret

# for differential operator matrix 

def generateFLForPNsByDim(pnlist,nodeTypeFaceLattice, HChain, TheChain,LChain, HDiff,LDiff,dim):
    """
    Generate the D_r and D_r+1 for a list of parameter node
    
    pnlist: list of parameter nodes index 
    
    nodeTypeFLatticList: facae lattice information (dictionary) for each atomic node type
    
    HChain: an index dict for C_r+1 chain element with values in integer
    TheChain: an index dict for C_r chain element with values in integer
    LChain: an index dict for C_r-1 chain element with values in integer
    
    HDiff: dict or sparse matrix for D: C_r+1 -> C_r
    LDiff: dict or sparse matrix for D: C_r -> C_r-1
    
    dim: r = dim for C_r
    
    return D_r=LDiff and D_r+1=HDiff
    """
    finishedTheChain = set()
    count =0
    for pn in pnlist:
        # add information of pn to HChain, TheChain,LChain, HDiff,LDiff
        print(count)
        count+=1
        generateFaceLatticeForSinglePNByDim(pn, nodeTypeFaceLattice, HChain, TheChain,LChain, HDiff,LDiff,finishedTheChain,dim)
    
    # no need to return, information is in HDiff, LDiff
    return 


def generateNodeTypeFLNoDim(nodeTypeFaceLattice):
    """
    The same as nodeTypeFaceLattice
    but without dimension information
    """
    nodeTypeFaceLatticeNoDim = []

    for i in range(len(nodeTypeFaceLattice)):
        #print(i)
        apnFL = nodeTypeFaceLattice[i]
        apnFLNoDim = dict()
        topDim = len(apnFL[list(apnFL.keys())[0]])-1
        for dim in range(1,topDim+1):
            #print(dim)
            for apn in apnFL:
                for cell in apnFL[apn][dim]:
                    if cell in apnFLNoDim:
                        continue
                    apnFLNoDim[cell] = set()
                    for face in apnFL[apn][dim-1]:
                        if set(face).issubset(set(cell)):
                            apnFLNoDim[cell].add(face)

        nodeTypeFaceLatticeNoDim.append(apnFLNoDim)
        
    return nodeTypeFaceLatticeNoDim
    

def addFaceInformationToDiffMatrix(c, nodeTypeFaceLatticeNoDim, cellIndex, chain ,diffMatrix):
    
    for i in range(len(c)):
        component = c[i]
        # 0-dim face is empty
        if len(component) == 1:
            continue
        apnFLNoDim = nodeTypeFaceLatticeNoDim[i]
        for componentFace in apnFLNoDim[component]:
            newface = list(c)
            newface[i] = componentFace
            newface = tuple(newface)
            if newface not in chain:
                chain[newface] = len(chain)
            diffMatrix[str((cellIndex, chain[newface]))] = 1
            
    return 

def generateFaceLatticeForSinglePNByDim(pn, nodeTypeFaceLattice, HChain, TheChain,LChain, HDiff,LDiff,finishedTheChain, dim):
    """
    Generate and add the information of parameter node to C_r-1, C_r, C_r+1, D_r D_r+1 
    
    pn: is the single parameter node
    
    nodeTypeFaceLattice: facae lattice information (dictionary) for each atomic node type
    
    HChain: an index dict for C_r+1 chain element with values in integer
    TheChain: an index dict for C_r chain element with values in integer
    LChain: an index dict for C_r-1 chain element with values in integer
    
    HDiff: dict or sparse matrix for D: C_r+1 -> C_r dimension of dim(C_r+1)*dim(C_r)
    LDiff: dict or sparse matrix for D: C_r -> C_r-1 dimension of dim(C_r)*dim(C_r-1)
    dim: r = dim for C_r
    """
    
    apnList = [nodeTypeFaceLattice[i][pn[i]] for i in range(len(pn))]
    
    pnHChain = tensorProductForDim(apnList,dim+1)
    pnTheChain = tensorProductForDim(apnList,dim)
    nodeTypeFaceLatticeNoDim = generateNodeTypeFLNoDim(nodeTypeFaceLattice)
    
    for c in pnHChain:
        if c not in HChain:
            HChain[c] = len(HChain)
            addFaceInformationToDiffMatrix(c, nodeTypeFaceLatticeNoDim, HChain[c],TheChain,HDiff)
            
    
    for c in pnTheChain:
        if c not in finishedTheChain:
            finishedTheChain.add(c)
            addFaceInformationToDiffMatrix(c, nodeTypeFaceLatticeNoDim, TheChain[c],LChain,LDiff)
    
        
    return 

def tensorProductForDim(atomicParameterNodeList,dim):
    upperBounds = [len(x)-1 for x in atomicParameterNodeList]
    topDim = sum(upperBounds)
   
    
    partition = integerPartition(dim,upperBounds)
    faces = set()
    for p in partition:
        #print(p)
        facelist = [atomicParameterNodeList[j][p[j]] for j in range(len(p))]
        faces |= set(product(*facelist))

        
    return faces

# suppose there are at most 2 thresholds
def stringifyParser(l):
    ret =  []
    l = eval(l)
    #print(l)
    for i in range(len(l)):
        inputNo = l[i][1][0]  
        m=outputNo = l[i][1][1] 
        polyNo = pow(2,inputNo)
        totalLen = polyNo*outputNo

        logic = l[i][1][2]
        logic = bin(int(logic,16))[2:].zfill(totalLen)

        lower = []
        middle = []
        upper = []
        thetas = l[i][2]
        temp = []
        for j in range(0,len(logic),m):
            polyIndex = polyNo - int(j/m) - 1
            piece = logic[j:j+m]
            #print(piece)
            if len(set(piece)) == 2:
                middle.append(polyIndex)
            elif set(piece) == set('1'):
                upper.append(polyIndex)
            else:
                lower.append(polyIndex)

        if len(thetas)==1:
            temp = sorted(lower) + [-thetas[0]-1] + sorted(upper)
        else:
            
            temp = sorted(lower) + [-thetas[0]-1] + sorted(middle) + [-thetas[1]-1]+sorted(upper)
            #print(temp)

        ret.append(tuple(temp))
    return ret


# following is the morse theory 

def checkCoreduction(face, coboundary, currComplex,used):
    for cell in coboundary:        
        if len(currComplex[cell] - used) == 1:
            return True
    return False


# used is the preset attractors
def genMatchingWithPreAtt(currComplex,coboundaryMatrix,used):
    
    kings = []
    queens = []
    aces = []
    
    count = 0
    while currComplex:
        count +=1
        
        # find queen, king pairs
        flag = True
        
        # flag: find coreducible pair in previous around
        while flag:
            tempComplex = dict()
            flag = False
            for cell in currComplex:

                tempDiff = currComplex[cell] - used

                if len(tempDiff) == 1:
                    queen = list(tempDiff)[0]
                    king = cell

                    queens.append(queen)
                    kings.append(king)

                    used.add(king)
                    used.add(queen)

                    flag = True

                if cell not in used:
                    tempComplex[cell] = tempDiff
            
            currComplex = tempComplex
            
        #print(queens, kings, aces)
        #print(tempComplex)
        
        # check coreducible or not
        flag = False # exist coreducible or not
        while not flag:
            tempComplex = dict()
            
            for key in currComplex:
                
                # if already have coreduction in previous loop, then only restore the rest of dictionary
                if flag:
                    tempComplex[key] = currComplex[key]
                    continue
                
                
                if len(currComplex[key]) == 0:
                    aces.append(key)
                    used.add(key)

                    # if key is a top cell, just continue
                    if key not in coboundaryMatrix:
                        continue

                    # if there is a coreducible pair we break the aces collection process
                    if checkCoreduction(key, coboundaryMatrix[key], currComplex, used):
                        flag = True
                        
                else:
                    tempComplex[key] = currComplex[key]
            
            currComplex = tempComplex
            
            # if empty
            if not currComplex:
                break
        
    return queens, kings, aces

def isFace(f,c):
    if set(f).issubset(set(c)):
        return True
    
    return False

def getBoundaryMatrix(apnList):
    faceLattice = dict()
    for index in apnList:
        pn = apnList[index]
        d = len(pn)
        for dim in range(d-1,0,-1):
            cells = pn[dim]
            faces = pn[dim-1]
            for c in cells:
                for f in faces:
                    if isFace(f,c):
                        sf = tuple(sorted(f))
                        sc = tuple(sorted(c))
                        if sc not in faceLattice:
                            faceLattice[sc] = []
                        faceLattice[sc].append(sf)
        
        # the zero diminsion
        cells = pn[0]
        for point in cells:
            faceLattice[tuple(sorted(point))] = []
            
            
    # remove the duplicate faces
    for key in faceLattice:
        faceLattice[key] = list(set(faceLattice[key]))
            
    return faceLattice

def getCoboundaryMatrix(boundaryMatrix):
    coboundaryMatrix = dict()
    
    for cell in boundaryMatrix:
        boundary = boundaryMatrix[cell]
        for b in boundary:
            if b not in coboundaryMatrix:
                coboundaryMatrix[b] = []
            coboundaryMatrix[b].append(cell)
    return coboundaryMatrix
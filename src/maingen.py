# Author: Answer03 

import cv2 
import numpy as np
import random

class musicpiece(object):
    def __init__(self):
        self.val=0

def reproduction(a,b):
    ret=musicpiece()
    ret.pcs=np.append(a.pcs[0:16],b.pcs[16:32])
    ret.dely=np.append(a.dely[0:16],b.pcs[16:32])
    prob=random.random()
    if prob<=0.05:
        pos=random.randint(0,31)
        ret.pcs[pos]=random.randint(0,26)
        ret.dely[pos] = random.randint(0,1)
    prob = random.random()
    return ret



harmy = [1.0,0.0,0.25,0.5,0.5,1.0,0.0,1.0,0.5,0.5,0.25,0.0]

def fitnes(piece1):
    cnt=0
    sum=0
    for i in range(31):
        if (piece1.pcs[i] != piece1.pcs[i+1]) or (piece1.dely[i]!=0):
            cnt=cnt+1
            dif=(piece1.pcs[i+1]-piece1.pcs[i]+36)%12
            sum=sum+1.0-harmy[dif]
    return sum/cnt

def cmp(pc1):
    return pc1.val

populrs = []
presumfit = []
sumfit=0
sumpopulrs=0

def randomselect():
    prob=random.random()*sumfit
    l=0
    r=sumpopulrs-1
    ans=0
    while l<=r:
        mid= (l+r)//2
        if prob <= presumfit[mid]:
            ans=mid
            r=mid-1
        else :
            l=mid+1
    return populrs[ans]


for i in range(100):
    prt1 = musicpiece()
    prt1.pcs=np.random.randint(0,high=27,size=32)
    prt1.dely=np.random.randint(0,high=2,size=32)
    prt1.val=fitnes(prt1)
    populrs.append(prt1)
    presumfit.append(0)
    sumpopulrs=sumpopulrs+1


Generations = 50
for tms in range(Generations):
    newpopulrs = []
    sumfit=0
    for i,va in enumerate(populrs):
        sumfit+= va.val
        if i==0:
            presumfit[i]=va.val
        else:
            presumfit[i]=presumfit[i-1]+va.val
    print(presumfit)
    for i in range(100):
        father = randomselect()
        mother = randomselect()
        newanm = reproduction(father,mother)
        newanm.val=fitnes(newanm)
        newpopulrs.append(newanm)

    populrs = newpopulrs

for i,va in enumerate(populrs):
    print(va.val)
    print(va.pcs)
    print(va.dely)

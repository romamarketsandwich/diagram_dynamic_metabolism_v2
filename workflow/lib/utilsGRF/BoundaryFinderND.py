import numpy as np
import matplotlib.pyplot as plt
import copy, os, sys
from scipy.optimize import fmin, fsolve
import itertools
import pandas as pd
import math

"""
    Code to compute boundaries of a model in an ND feature space.
    Copyright (C) <2021>  <Rosa Martinez Corral>
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """

class BoundaryExplorer():
    def __init__(self,compute_x_y_f=None,pars_limit=[1e-5,1e5],constraints=None,npars=0,wpindices=[None,None],cg0=-1,nsites=0,arrays=None,seed=1,dfpt=None,tol_difference_pars=1e-8,ratios_indices=[None]):
        """This is a Class to found the boundaries of a model in ND parameter space. For historic reasons it uses position/steepness for the variable notation and so on, but it  works for any ND feature space. For efficiency reasons, it should probably not be used for more than 5 or 6 dimensions. As of now it is a very poor and nonefficient implementation but is helpful to get a rough idea of the model behaviour.
        By starting randomly from a few points, it will iteratively change them in various ways in order to find new points that expand the boundary.
        It uses a grid, and fills that grid. It outputs a dataframe with the results (or more than one to avoid excessive memory).

        compute_x_y_f: function that takes an array of parameter values as argument and returns the corresponding point in 2D space. 
        pars_limit: minumum and maximum values of the parameter values, in general.
        constraints:dictionary with constraints on the parameter values. 
        The keys of the dictionary are the indices of the parameters, and value is a dictionary with various keywords:
        - 'min', 'max': min and max values for that parameter if different than pars_limit
        - 'fixed': set to 1 if this parameter is to be kept fixed. Use together with 'min' AND 'max' to specify the value (same for min and max)
        - 'target': parameter that is taken as a reference, useful for instance if want to keep to parameters to the same values, or within a range. Use with either:
         > 'fcd' and 'fcu' to specify the fold-change with respect to this target to which the parameter value is to be kept.
         > 'gt' or 'lt' if the parameter value should be grater or smaller than the target.
        -npars: number of parameter values
        - row_ar: row grid limits
        - col_ar: col grid limits
        - seed: seed for the random number generator 
        - 
        - 
        - tol_difference_pars: minimum difference between two parameter sets that is relevant to consider that they are different. 
        -wpindices: for the polymerase model explored with activation effect, first and last index of the polymerase cooperativities. 
        -nsites: also for the polymerase model, number of binding sites
        -cg0: index of the first parameter that corresponds to transition between two conformations, for the coarse grained models (allostery)
        -ratios_indices: to be used when we want to constrain certain label ratios with respect to other label ratios. 
            List of lists of 3 elements:
                First element contains the indices of forward and reverse transitions of the ratio of interest. 
                Second element contains another list of lists. Each list contains the indices of the forward and reverse transitions of the ratios that serve as reference.
                The third element contains the number of binding sites occupied for the transition of interest.

        """



        self.npars=npars #number of total parameters
        self.tol_difference_pars=tol_difference_pars
        self.arrays=arrays #array defining the row bins of the grid
        self.max_arrays=[x[-1] for x in arrays]
        self.maxbin_arrays=[x[-2] for x in arrays]
        self.min_arrays=[x[0] for x in arrays]
        self.ND=len(arrays)
        otherddict=dict()
        for x in range(self.ND):
            otherd=[y for y in range(self.ND) if y!=x]
            otherddict[x]=otherd
        self.otherddict=otherddict

        self.cg0=cg0
        if self.cg0>0:

            #check that none of the parameters greater or equal to this one is in hte constraint dictionary with target. Not implemented
            if constraints is not None:
                for key in constraints.keys():
                    if constraints[key]>=self.cg:
                        subdict=constraints[paridx]
                        subkeys=subdict.keys()
                        if 'fixed' in subkeys:
                            if subdict['fixed']>0:
                                raise ValueError("Code is not prepared to handle fixed transitions between conformations while keeping conformations in the range.")
                        if 'target' in subkeys:
                                raise ValueError("Code is not prepared to handle transitions between conformations with a target while keeping conformations in the range.")




        if wpindices[0] is not None: #Equilibrium polymerase model specific

            self.sortedidxs=wpindices #first and last value of the chunk of parameters that correspond to wp
            self.polstrlist=[]
            for l in range(1,nsites+1):
                for x in itertools.combinations(range(1,nsites+1),l):
                    self.polstrlist.append('wp'+','.join(['%s'%site for site in x]))
            
            idx0=wpindices[0]
            reference_list=[]
            for i in range(len(self.polstrlist)):
                wpname=self.polstrlist[i]
                sites=wpname.split('p')[1].split(',') #find which names have at least one of the sites
                ns=len(sites)
                otheridxs=[]
                if ns>1:
                    if ns==2:
                        combis=sites
                    else:
                        combis_=itertools.combinations(sites,ns-1)
                        combis=[]
                        for x in combis_:
                            combis.append(','.join(sorted(x)))
                    for j in range(0,i):
                        sitesprev=self.polstrlist[j].split('p')[1]
                        if sitesprev in combis:
                            otheridxs.append(idx0+j) #has to be the index with respect to the whole parameter set
                reference_list.append(list(set(otheridxs)))

            self.previouswplist=reference_list
            
        else:
            self.sortedidxs=[None,None] #this will have no effect then and will just explore as usual
            self.polstrlist=None
            self.previouswplist=None

        if ratios_indices[0] is not None:
            self.ratios_indices_lists=sorted(ratios_indices,key=lambda x: x[2])
            #self.ratios_indices=[]

            #for out in self.ratios_indices_lists:
            #    self.ratios_indices.extend(out[0]) #add indices of numerator of parameters whose values depend upon previous ones
        else:
            self.ratios_indices_lists=[]

        constraint_mat=np.zeros((npars,12)) #0: min, 1: max, 2: log10min, 3: log10max,4: target idx, 5: fc down if target, 6: fc up if target,7: log10fcd, 8: log10fu, 9: smaller or equal than target, 10: greater or equal than target, 11: fixed (negative value: false. positive: true). 
        constraint_mat[:,0]=pars_limit[0]
        constraint_mat[:,1]=pars_limit[1]
        constraint_mat[:,2]=np.log10(pars_limit[0])
        constraint_mat[:,3]=np.log10(pars_limit[1])
        constraint_mat[:,4:]=-1
        if constraints is not None:
            for paridx in constraints.keys():
                      

                subdict=constraints[paridx]
                subkeys=subdict.keys()
                if 'fixed' in subkeys:
                    if subdict['fixed']>0:
                        
                        if np.abs(subdict['min']-subdict['max'])>0.00000001:
                            print('Fixed parameter value wrongly defined for idx %d. Exiting.'%paridx)
                            sys.exit()
                        constraint_mat[paridx][0]=subdict['min']
                        constraint_mat[paridx][-1]=1
                else:
                    if 'min' in subkeys:
                        constraint_mat[paridx][0]=subdict['min']
                        constraint_mat[paridx][2]=np.log10(subdict['min'])
                    if 'max' in subkeys:
                        constraint_mat[paridx][1]=subdict['max']
                        constraint_mat[paridx][3]=np.log10(subdict['max'])
                    if 'target' in subkeys:

                        if isinstance(subdict['target'],int):
                            constraint_mat[paridx][4]=subdict['target']
                            if 'fcd' in subkeys or 'fcu' in subkeys:
                                #print('Dependency constraint is wrongly specified. Exiting.')
                                #sys.exit()
                                constraint_mat[paridx][5]=subdict['fcd']
                                constraint_mat[paridx][6]=subdict['fcu']
                                constraint_mat[paridx][7]=np.log10(subdict['fcd'])
                                constraint_mat[paridx][8]=np.log10(subdict['fcu'])
                            else:
                                gt=int(-1)
                                lt=int(-1)
                                if 'gt' in subkeys:
                                    gt=int(subdict['gt'])
                                if 'lt' in subkeys:
                                    lt=int(subdict['lt'])

                                if gt+lt==0:
                                    if lt>0:
                                        constraint_mat[paridx][9]=1
                                    if gt>0:
                                        constraint_mat[paridx][10]=1
                                else:
                                    print('gt or lt is wrongly specified. Exiting')
                                    sys.exit()
                        else: #it depends on multiple targets
                            constraint_mat[paridx]=self.npars+1 #to indicate that I have to look at self.multiple_constraints





        self.constraints=constraint_mat
        self.multiple_constraints=constraints
        
        self.compute_x_y=compute_x_y_f
        
        ND=self.ND
        if dfpt is None: 
            self.dfpt=pd.DataFrame()
            colsgrow=[]
            #columns to hold the point values. Each column is a coordinate in ND space
            for i in range(ND):
                self.dfpt[i]=0 #this will contain the array bin corresponding to this dimension
            #columns to hold info about whether point is at a boundary to go with less or more values
            for i in range(ND): #iterate twice so that first I have the points
                name1='%d_d'%(i)
                name2='%d_u'%(i)
                self.dfpt[name1]=0
                self.dfpt[name2]=0
                colsgrow.append(name1)
                colsgrow.append(name2)

            #columns to hold the parameter values
            for i in range(npars):
                self.dfpt['p%d'%i]=0
            
            self.emptymutlist=[0]*len(colsgrow)
            
            self.colsgrow=colsgrow
        else:
            self.dfpt=dfpt
                 
        
    
        np.random.seed(seed)
        
            
    def get_num_den_fromratio(self,numerator,denominator,numval,denval,maxprevr):
        #randomly choose numerator or denominator to have a random value. This will avoid bias in either choosing one or another always
        if np.random.uniform(0,1)<0.5:
            #numerator as a function of the rest
            minval,maxval, minlog10,maxlog10=self.constraints[numerator][0:4]
            minvalnew=denval*maxprevr
            if minvalnew>maxval:

                #in this case put the numerator to the maxval and change denominator to a smaller number
                numvalnew=maxval
                minvald,maxvald, minlog10d,maxlog10d=self.constraints[denominator][0:4]
                maxvald2=numvalnew/maxprevr
                if maxvald2<=minvald:
                    denvalnew=minvald
                else:
                    denvalnew=10**np.random.uniform(minlog10d,np.log10(maxvald2))
                
            else:
                numvalnew=10**np.random.uniform(np.log10(minvalnew),maxlog10)
                denvalnew=denval
        else:
            #denominator as a function of the rest
            minvald,maxvald, minlog10d,maxlog10d=self.constraints[denominator][0:4]
            maxvalnew=numval/maxprevr
            if maxvalnew<minvald:
                #put denominator to minval and change numerator to a larger value
                denvalnew=minvald
                minval,maxval, minlog10,maxlog10=self.constraints[numerator][0:4]
                minval2=denvalnew*maxprevr
                if minval2>=maxval:
                    numvalnew=minval2
                else:
                    numvalnew=10**np.random.uniform(np.log10(minval2),maxlog10)
            else:
                denvalnew=10**np.random.uniform(minlog10d,np.log10(maxvalnew))
                numvalnew=numval
        if (numvalnew/denvalnew)<maxprevr:
            print(numvalnew,denvalnew, numvalnew/denvalnew, "is smaller than", maxprevr)
            raise ValueError
        else:
            return [numvalnew,denvalnew]
    
    

    def check_point_inrange(self,point):
        for d in range(len(point)):
            if point[d]<self.min_arrays[d] or point[d]>=self.max_arrays[d]:
                return False
        return True
    def save_point(self,point,pars,prob_replace=0,checkb=False):
        if not checkb:
            to_append=list(point)+self.emptymutlist+list(pars)
            a_series = pd.Series(to_append, index = self.dfpt.columns)
            self.dfpt = self.dfpt.append(a_series, ignore_index=True)
            self.newpoints+=1
            return True
        else:
            update=[]
            for d in range(self.ND): #for all other dimensions fixed, see if this point is at the boundary
                nond=[x for x in range(self.ND) if x!=d]
                masks=[(self.dfpt[x].values==point[x]) for x in nond]
                line=self.dfpt.iloc[list(np.logical_and(*masks))] #print(masks)
                #print(line)
                if len(line)>0: #points that share all dimensions except dimension d with point 
                    

                    #if newpt[d]<min(line[:,d]
                    argmin=np.argmin(line[d].values)
                    
                    argmax=np.argmax(line[d].values)
                    minval=line[d].values[argmin]
                    maxval=line[d].values[argmax]
                    if point[d]<minval: 
                        update.append(1)
                    elif point[d]==minval:
                        if np.random.uniform(0,1) < prob_replace:
                            update.append(1)
                        else:
                            update.append(0)
                    else:
                        update.append(0)

                    if point[d]>maxval:                   
                        update.append(1)
                    elif point[d]==maxval:
                        if np.random.uniform(0,1) < prob_replace:                                        
                            update.append(1)
                        else:
                            update.append(0)
                    else:
                        update.append(0)
                else:
                    update.extend([1,1]) #it is at the boundary since no other point shares the other dimensions

            if sum(update)>0:
                #idx=len(self.dfpt)
                #self.dfpt.iloc[idx]=list(point)+update+list(pars)
                to_append=list(point)+update+list(pars)
                a_series = pd.Series(to_append, index = self.dfpt.columns)
                self.dfpt = self.dfpt.append(a_series, ignore_index=True)
                self.newpoints+=1

            return 


    def get_initial_points(self,ninitpoints):
        #Generate ninitpoints to begin with
        
        self.newpoints=0
        
        while self.newpoints < ninitpoints:
            pars=np.zeros(self.npars)

            #First generate the free parameters
            for pnum in range(self.npars):
                if self.sortedidxs[0] is not None:
                    if pnum>=self.sortedidxs[0] and pnum<=self.sortedidxs[1]: #if it is within the parameters that have to maintain a given order
                        idx=pnum-self.sortedidxs[0] #which wp is, starting from 0 #INITIALISE
                        wpname=self.polstrlist[idx] #INITIALISE. GENERATE DYNAMICALLY AT THE BEGINNING, FROM A PARAMEER WITH THE NUMBER OF SITES
                        previouswp=self.previouswplist[idx] #INITIALISE
                        if len(previouswp)<1: #means it is the individual affinity between the site and polymerase
                            free=True
                            #print('individual',wpname)
                        else:
                            free=False
                    else:
                        if (self.constraints[pnum][4]<0): #does not depend upon target
                            free=True
                        else:
                            free=False 


                else:
                    if (self.constraints[pnum][4]<0): # and pnum not in self.ratios_indices: #does not depend upon target
                        free=True
                    else:
                        free=False

                if free:
                    if self.cg0>0 and pnum>=self.cg0:
                            minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                            if pnum==self.cg0:
                                pars[pnum]=10**np.random.uniform(minlog10,maxlog10)
                            else:
                                allprev=pars[self.cg0:pnum]
                                minprev=np.min(allprev)
                                maxprev=np.max(allprev)
                                if minprev<1:
                                    maxlog10=np.log10(maxval*minprev)
                                if maxprev>1:
                                    minlog10=np.log10(minval*maxprev)
                                pars[pnum]=10**np.random.uniform(minlog10,maxlog10)

                    else:

                        if self.constraints[pnum][-1]>0: #fixed par
                            pars[pnum]=self.constraints[pnum][0]
                            #print('Fixing par', pnum, 'to ', pars[pnum])
                        else: #if pnum not in self.ratios_indices:
                            minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                            pars[pnum]=10**np.random.uniform(minlog10,maxlog10)

            #Then go through the constraint ones with respect to a given target, which will have been set in the lines above
            pars_with_c=np.where(self.constraints[:,4]>-1)[0]
            if len(pars_with_c)>0:
                for pnum in pars_with_c:
                    targetidx,fcd,fcu=self.constraints[pnum][4:7]
                    minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                    lt,gt=self.constraints[pnum][9:11]
                    if targetidx<self.npars:
                        targetvalue=pars[int(targetidx)]
                        if lt<0 and gt<0:
                            min_=np.log10(max(minval,targetvalue*fcd))
                            max_=np.log10(min(maxval,targetvalue*fcu))
                        elif lt>0:
                            max_=np.log10(targetvalue)
                            min_=minlog10
                        elif gt>0:
                            max_=maxlog10
                            min_=np.log10(targetvalue)
                        if np.abs(min_-max_)<0.0001:#equal:
                            pars[pnum]=10**min_
                        else:
                            pars[pnum]=10**(np.random.uniform(min_,max_))
                    else:#depends upon multiple targets
                        targetvalues=self.multiple_constraints[pnum]['target']
                        current_min=minval
                        current_max=maxval
                        for target in targetvalues:
                            subdict=self.multiple_constraints[pnum][target]
                            subkeys=subdict.keys()
                            if 'fcd' in subkeys:
                                current_min=max(current_min,pars[int(target)]*fcd)
                            if 'fcu' in subkeys:
                                current_max=min(current_max,pars[int(target)]*fcu)
                                
                            
                            if 'gt' in subkeys:
                                current_min=max(current_min,pars[int(target)])
                            if 'lt' in subkeys:
                                current_max=min(current_max,pars[int(target)])
                        min_=np.log10(current_min)
                        max_=np.log10(current_max)
                        if np.abs(min_-max_)<0.0001:#equal:
                            pars[pnum]=10**min_
                        else:
                            pars[pnum]=10**(np.random.uniform(min_,max_))


                    
                    #print('target',targetvalue,'new',pars[pnum], targetvalue/pars[pnum])
            
            #then go through the cooperativities that have to be ordered
            #Equilibrium polymerase model
            if self.polstrlist is not None:
                for pnum, par in enumerate(pars):
                    if pnum>=self.sortedidxs[0] and pnum<=self.sortedidxs[1]:
                        idx=pnum-self.sortedidxs[0] #which wp is, starting from 0 #INITIALISE
                        wpname=self.polstrlist[idx] #INITIALISE. GENERATE DYNAMICALLY AT THE BEGINNING, FROM A PARAMEER WITH THE NUMBER OF SITES
                        previouswp=self.previouswplist[idx] #INITIALISE
                        if len(previouswp)>0: 
                            #Now get which is the corresponding lower bound according to the other cooperativities, which sets a lower bound
                            lowerbound=np.max(pars[previouswp])
                            #find the previous level values with the sites in wpname, and find a larger or equal parameter value
                            minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                            
                            newp=10**np.random.uniform(np.log10(lowerbound), maxlog10)                        

                            pars[pnum]=newp

            #Nonequilibrium polymerase model
            if len(self.ratios_indices_lists)>0:
                for out in self.ratios_indices_lists:
                    numerator,denominator=out[0]
                    #First I assign them a random value
                    numval=pars[numerator]
                    denval=pars[denominator]
                    prev_rs=[]
                    for r2 in out[1]:
                        rval=pars[r2[0]]/pars[r2[1]]
                        prev_rs.append(rval)
                    maxprevr=max(prev_rs)
                    #print(numerator,denominator,numval,denval,maxprevr)

                    numvalnew,denvalnew=self.get_num_den_fromratio(numerator,denominator,numval,denval,maxprevr)

                    pars[numerator]=numvalnew
                    pars[denominator]=denvalnew


            point=self.compute_x_y(pars)
            #p,s=compute_pos_s_inputoutput(self.xrange,pars,self.ssfunc)
            if point[0] is not None:
                inrange=self.check_point_inrange(point)
                if inrange:
                    point2=[]
                    for d in range(len(point)):
                        point2.append(np.where(point[d]>=self.arrays[d])[0][-1])
                    self.save_point(point2,pars)
                
                        
    def find_boundary_points(self):
        """
        
        """
        df=self.dfpt
        for d in range(self.ND):
            otherd=self.otherddict[d]
            for line_ in df.groupby(otherd): #look at the points that share all dimensions except the one analysed
                line=line_[1] #this is a dataframe
                if len(line)==1:
                    if line[d].values[0]==self.min_arrays[d]: 
                        vald=2 #cannot grow more towards lower values
                    else:
                        vald=1
                    if line[d].values[0]==self.maxbin_arrays[d]:
                        valu=2 #cannot grow more towards higher values
                    else:
                        valu=1
                    df.loc[line.index[0],'%d_d'%d]=vald
                    df.loc[line.index[0],'%d_u'%d]=valu
                else:
                    argmin=np.argmin(line[d].values)
                    argmax=np.argmax(line[d].values)
                    #print(line.index)
                    #print(line)
                    #print(line.index[argmin])
                    #print(line.index[argmax])
                    if line[d].values[argmin]==self.min_arrays[d]:
                        minv=2
                    else:
                        minv=1
                    if line[d].values[argmax]==self.maxbin_arrays[d]:
                        maxv=2
                    else:
                        maxv=1

                    df.loc[line.index[argmin],'%d_d'%d]=minv
                    df.loc[line.index[argmax],'%d_u'%d]=maxv
        
    
    
    def mutate_parset(self,parset,prob_par=1,trials=20,extr_uniform=[-1,1],prob_replace=0,targetted=False):
        #modifies a given parameter set to find a new one that lies on the boundary
        #this must be extensively tested to find the best way
        #there could also be different functions 
        done=False
        factor=1
        uniform=True
        fmin,fmax=extr_uniform
        for_targetted=False
        
        ntr=0
       
        while (not done) and (ntr < trials):
            pars=parset.copy() #make a copy to mutate parameter set
            ntr+=1
            #select how the new parameter is chosen
            #print(trials,sep=',')
            if ntr>2:
                if ntr < 6:
                    uniform=False
                    factor=0.1
                elif ntr < 10:
                    uniform=False
                    factor=0.5
                elif ntr < 14:
                    uniform=False
                    factor=1
                elif ntr < 16:
                    uniform=False
                    factor=2
                else:
                    uniform=True
                    factor=0.5
                    
            #mutate each parameter
            #First go through the free ones:
            
            for pnum, par in enumerate(parset):
                if self.sortedidxs[0] is not None:
                    if pnum>=self.sortedidxs[0] and pnum<=self.sortedidxs[1]:
                        idx=pnum-self.sortedidxs[0] #which wp is, starting from 0 #INITIALISE
                        wpname=self.polstrlist[idx] #INITIALISE. GENERATE DYNAMICALLY AT THE BEGINNING, FROM A PARAMEER WITH THE NUMBER OF SITES
                        previouswp=self.previouswplist[idx] #INITIALISE
                        if len(previouswp)<1: #means it is the individual affinity between the site and polymerase
                            free=True
                            #print('individual',wpname)
                        else:
                            free=False
                    else:
                        if (self.constraints[pnum][4]<0): #does not depend upon target
                            free=True
                        else:
                            free=False 

                else:
                    free=True

                if free:

                    if not (( self.constraints[pnum][-1]>0) or (self.constraints[pnum][4]>-1)): #not a fixed parameter nor parameter whose value depends upon another
                        if np.random.uniform(0,1) < prob_par:
                            minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                            if self.cg0>0 and pnum>self.cg0:
                                allprev=pars[self.cg0:pnum]
                                minprev=np.min(allprev)
                                maxprev=np.max(allprev)
                                if minprev<1:
                                    #print("prev value is", prev, maxval)
                                    maxval=(maxval*minprev)
                                    #print("new maxval", maxval)
                                if maxprev>1:
                                    minval=(minval*maxprev)
                                
                            if uniform is True:
                            #if pnum in indices:
                                newp=par*(10**np.random.uniform(fmin/factor,fmax/factor))                            
                            else:
                                newp=np.random.normal(par,par/factor)
                            
                            if newp<minval:
                                newp=minval
                            elif newp>maxval:
                                newp=maxval
                            pars[pnum]=newp
                        elif self.cg0>0 and pnum>self.cg0: #for the transitions between conformations for the coarse grained parameters, make sure that it is still in the range
                            minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                            allprev=pars[self.cg0:pnum]
                            minprev=np.min(allprev)
                            maxprev=np.max(allprev)
                            if minprev<1:
                                #print("prev value is", prev, maxval)
                                maxval=(maxval*minprev)
                                #print("new maxval", maxval)
                            if maxprev>1:
                                minval=(minval*maxprev)
                            if par>maxval:
                                pars[pnum]=maxval
                            elif par<minval:
                                pars[pnum]=minval



            #then go through the constraint ones with respect to a target. 
            pars_with_c=np.where(self.constraints[:,4]>-1)[0] #Parameters which are referenced with respect to a target
            if len(pars_with_c)>0:
                for pnum in pars_with_c:
                    par=pars[pnum]
                    targetidx,fcd,fcu=self.constraints[pnum][4:7]
                    minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                    lt,gt=self.constraints[pnum][9:11]
                    if targetidx<self.npars:
                        targetvalue=pars[int(targetidx)]
                        if lt<0 and gt<0:
                            min_=np.log10(max(minval,targetvalue*fcd))
                            max_=np.log10(min(maxval,targetvalue*fcu))
                        elif lt>0: #smaller than the target
                            max_=np.log10(targetvalue)
                            min_=minlog10
                        elif gt>0: #greater than the target
                            max_=maxlog10
                            min_=np.log10(targetvalue)
                        
                        #if np.abs(min_-max_)<0.0001:#equal:
                        #    newp=10**min_
                        #else:
                        min10=10**min_
                        max10=10**max_
                    else:#depends upon multiple targets
                        targetvalues=self.multiple_constraints[pnum]['target']
                        current_min=minval
                        current_max=maxval
                        for target in targetvalues:
                            subdict=self.multiple_constraints[pnum][target]
                            subkeys=subdict.keys()
                            if 'fcd' in subkeys:
                                current_min=max(current_min,pars[int(target)]*fcd)
                            if 'fcu' in subkeys:
                                current_max=min(current_max,pars[int(target)]*fcu)
                                
                            
                            if 'gt' in subkeys:
                                current_min=max(current_min,pars[int(target)])
                            if 'lt' in subkeys:
                                current_max=min(current_max,pars[int(target)])
                        min_=np.log10(current_min)
                        max_=np.log10(current_max)
                        
                        min10=current_min
                        max10=current_max
                            
                    if np.random.uniform(0,1) > prob_par:
                        #don't mutate it randomly but make sure that it is equal to the reference cooperativities
                        newp=par
                        if newp<min10:
                            newp=min10
                        elif newp>max10:
                            newp=max10

                    else:

                        if uniform is True:
                        #if pnum in indices:
                            newp=par*(10**np.random.uniform(fmin/factor,fmax/factor))
                        else:
                            #if lt<0 and gt<0:
                            #    newp=np.random.normal(targetvalue,targetvalue/factor)
                            #else:
                            newp=np.random.normal(par,par/factor)
                            
                        if newp<min10:
                            newp=min10
                        elif newp>max10:
                            newp=max10
                    
                    pars[pnum]=newp

            #Equilibrium polymerase model parameters
            if self.polstrlist is not None:
                for pnum, par in enumerate(parset):
                    if pnum>=self.sortedidxs[0] and pnum<=self.sortedidxs[1]:
                        idx=pnum-self.sortedidxs[0] #which wp is, starting from 0 #INITIALISE
                        wpname=self.polstrlist[idx] #INITIALISE. GENERATE DYNAMICALLY AT THE BEGINNING, FROM A PARAMEER WITH THE NUMBER OF SITES
                        previouswp=self.previouswplist[idx] #INITIALISE
                        if len(previouswp)>0: #means it is the individual affinity between the site and polymerase
                            #Now get which is the corresponding lower bound according to the other cooperativities, which sets a lower bound
                            lowerbound=np.max(pars[previouswp])
                            #print('constraint wp',wpname, lowerbound)
                            #print(pars[previouswp])
                            if np.random.uniform(0,1) > prob_par:
                                #don't mutate it randomly but make sure that it is equal to the reference cooperativities
                                newp=par
                                if newp<lowerbound:
                                    newp=lowerbound
                                
                            else:
                                #find the previous level values with the sites in wpname, and find a larger or equal parameter value
                                minval,maxval, minlog10,maxlog10=self.constraints[pnum][0:4]
                                if uniform is True:
                                #if pnum in indices:
                                    newp=par*(10**np.random.uniform(fmin/factor,fmax/factor))                            
                                else:
                                    newp=np.random.normal(par,par/factor)

                                if newp<lowerbound:
                                    newp=lowerbound
                                elif newp>maxval:
                                    newp=maxval
                            pars[pnum]=newp
                        #print('new value is',newp)
            #Nonequilibrium pol model parameterrs
            #Nonequilibrium polymerase model
            if len(self.ratios_indices_lists)>0:
                for out in self.ratios_indices_lists:
                    numerator,denominator=out[0]
                    #First I assign them a random value
                    numval=pars[numerator]
                    denval=pars[denominator]
                    prev_rs=[]
                    for r2 in out[1]:
                        rval=pars[r2[0]]/pars[r2[1]]
                        prev_rs.append(rval)
                    maxprevr=max(prev_rs)

                    numvalnew,denvalnew=self.get_num_den_fromratio(numerator,denominator,numval,denval,maxprevr)

                    pars[numerator]=numvalnew
                    pars[denominator]=denvalnew

            
            if np.any(np.abs(pars-parset)>self.tol_difference_pars):#if at least one parameter is different
        
                point=self.compute_x_y(pars)
                #p,s=compute_pos_s_inputoutput(self.xrange,pars,self.ssfunc)

                if point[0] is not None:
                    inrange=self.check_point_inrange(point) #False if for any of the dimensions is outside the range, true otherwise
                    
                    if inrange:
                        
                        if targetted:
                            for_targetted=True
                        else:
                            #save if it is a new boundary point
                            #flag=False
                            point2=[]
                            for d in range(len(point)):
                                point2.append(np.where(point[d]>=self.arrays[d])[0][-1])

                            self.save_point(point2,pars,checkb=True,prob_replace=prob_replace)
                        
            
        if targetted:
            if for_targetted:
                return [point,pars]
            else:
                return [[None],None]

    def approach_target(self,niters_target,parset,point1,point2,extr_uniform=None,prob_par=1,prob_replace=0,tolerance=0.01):
        #will try to approach a given target position and steepness
        point1=np.asarray(point1)
        point2=np.asarray(point2)
        d0=np.sum((point2-point1)**2) #initial distance. I'm not taking the sqrt so that I avoid operation
        
        
        distances=np.zeros(niters_target)
        distances[0]=d0
       
        bestp=point1
        
        i=1
        converged=False
        while i < niters_target and converged == False:
            d=-1
           
            point,pars=self.mutate_parset(parset,trials=1,prob_par=prob_par,prob_replace=prob_replace,extr_uniform=extr_uniform,targetted=True)
             
            if point[0] is not None:
                point=np.asarray(point)
                d=np.sum((point2-point)**2)
               

            if d>0 and d<distances[i-1]:
                parset=pars #new pars are these ones
                bestp=point
                distances[i]=d
            else:
                distances[i]=distances[i-1] #to avoid increasing
                  
            if d>-1 and d < tolerance: #it is sufficiently close to the target. added d>-1
                converged=True
            elif i > 20:
                if np.all(np.diff(distances[i-20:i]))<tolerance/2: #no improvement done
                    converged=True
            i+=1
            
        #check if a new boundary point has been found and modify self.mat and self.mat_pars accordingly
        inrange=self.check_point_inrange(bestp) #False if for any of the dimensions is outside the range, true otherwise 
        if inrange:

            point2=[]
            for d in range(len(bestp)):
                point2.append(np.where(bestp[d]>=self.arrays[d])[0][-1])
            self.save_point(point2,parset,checkb=True,prob_replace=prob_replace)

    
    def plot_boundary(self,title):
        ncol=4
        ND=self.ND
        npairs=math.factorial(ND)/(math.factorial(ND-2)*math.factorial(2))
        nrow=int(np.ceil(npairs/ncol))
        fig,axes=plt.subplots(nrow,ncol,figsize=(3*ncol,3*nrow))
        n_=0
        for d1,dim1 in enumerate(range(ND-1)):
            for d2,dim2 in enumerate(range(dim1+1,ND)):
                ax=axes[n_//ncol][n_%ncol]
                #print(max(self.arrays[dim1]),max(self.dfpt[dim1].values))
                #print(max(self.arrays[dim2]),max(self.dfpt[dim1].values))

                ax.scatter(self.arrays[dim1][list(map(int,self.dfpt[dim1].values))],self.arrays[dim2][list(map(int,self.dfpt[dim2].values))])
                ax.set_xlim(self.arrays[dim1][0],self.arrays[dim1][-1])
                ax.set_ylim(self.arrays[dim2][0],self.arrays[dim2][-1])
                ax.set_xlabel('%d'%dim1)
                ax.set_ylabel('%d'%dim2)
                n_+=1
        plt.tight_layout()
        plt.show()

    def savedf(self,name):
        dfname=os.path.join(self.folder_save,'df_'+name+'.csv')
        self.dfpt.to_csv(dfname)
        #print("saved", dfname)
        if self.eraseintermediates is True:
            if self.prevdfname is not None:
                #delete previous
                #print("removing", self.prevdfname)
                os.remove(self.prevdfname)
                
            self.prevdfname=dfname

    def extend_boundary(self,dofirstmutate=True,radius=1, maxradius=10, dopullcentroids=True,
        prob_par=1,prob_replace=0,extr_uniform=[-1,1],L_project=10,tol_target=0.001,
        niters=100, niters_conv=10,niters_conv_points=10,niters_target=500, 
        niters_save=30, folder_save='./',name_save='out', 
        plotting=False,verbose=False,eraseintermediates=True):


        """Function that extends the current boundary. Arguments:
        - dofirstmutate: modifies parameter values without attempting to approach any target. 
        - dopulltangents: when modifying, tries to approach a target point at an angle theta with respect to the line that links the point and a neighbor. 
        - dopullcentroids: when modifying, tires to approach a point in the direction of the line that links the current point with the centroid. 
        (Any of the dofirstmutate, dopulltangents, dopullcentroids set to True will be executed, in a consecutive manner )
        - radius: number of times a parameter is attempted to be mutated to begin with
        - maxradius: maximum number of times parameters will be attempted to be mutated
        - prob_par: probability that a parameter is selected to be mutated (between 0 and 1)
        - prob_replace: probability that a parameter set that occupies a given grid cell will be replaced if a new parameter set is found to lie on the same grid cell. 
        - extr_uniform: interval size in log space around the parameter value attempted to mutation. [-1,1] means a new parameter value is chosen in the range 0.1 x value- 10 x value
        - L_project: distance to the target point when doing pulling (in units of grid cells)
        - tol_target: distance to decide it is close enough and stop pulling towards a target.
        - niters: number of maximum iterations
        - niters_conv: number of iterations with no change in the boundary to decide that it has converged and stop the search. 
        - niters_conv_points: number of iterations a given point is tried to mutate before it is not tried again. 
        - niters_target: number of times it tries mutating to approach the target
        - niters_save: output interval
        - folder_save: folder where the output should be saved
        - name_save: name of the files for mat and mat_pars (to which either 'mat_' or 'mat_pars' will be preapended)
        - plotting: boolean to indicate whether to plot or not
        - verbose: boolean to indicate if messages should be printed.
        """

        self.find_boundary_points()
        self.prevdfname=None
        self.converged=False
        self.folder_save=folder_save
        
        i=0
        ND=self.ND

        self.nboundary=np.zeros(niters)
        self.nboundary[0]=len([x for x in self.dfpt[ND:ND+2*ND].values if np.any(x)==1])
        self.extr_uniform=extr_uniform
        self.radius=radius
        self.eraseintermediates=eraseintermediates
        
        while i < niters and self.converged== False:

            centroid=np.mean(self.dfpt.values[:,:ND],axis=0)
            if plotting:
                self.plot_boundary('i=%d'%i)


            self.newpoints=0

            if i%niters_save ==0:

                name='all_%s_%d'%(name_save,i)
                self.savedf(name)
                

            print(i,end=",",flush=True)
            #this could be made to depend upon the iteration
            #fmin=self.extr_uniform[0]
            #fmax=self.extr_uniform[1]
            
            if dofirstmutate:
                print("mutate")

                if i > 2:
                    if np.sum(self.nboundary[i-2:i])<1: #when it seems that it starts stopping, try more mutations per parameter
                        if self.radius+1 <= maxradius:
                            self.radius=self.radius+1
                
                for pti in range(len(self.dfpt)):
                    if pti%1000 ==0:

                        name='mut_%s_%d'%(name_save,pti)
                        self.savedf(name)
                    #print(pti,end=",")
    
                    row=self.dfpt.iloc[pti].values
                    if np.any(row[ND:ND+2*ND]==1):#it can grow
                        #for each of the dimensions where it can grow, pull in that direction as long as it means it is getting away from center of mass
                        point=row[0:ND]
                        parset=row[ND+2*ND:]
                        for j in range(self.radius):#radius number of times 
                            self.mutate_parset(parset,prob_par=prob_par,prob_replace=prob_replace,extr_uniform=extr_uniform)
                
                        
                self.find_boundary_points()
                #newnboundary=len([x for x in self.dfpt[ND:ND+2*ND].values if np.any(x)==1])
                self.nboundary[i]=self.newpoints

                if plotting:
                    self.plot_boundary('first m,i=%d'%i)
            
            if dopullcentroids:
                #print("pull")
                #print(len(self.dfpt))

                for pti in range(len(self.dfpt)):
                    if pti%1000 ==0:

                        name='pull_%s_%d'%(name_save,pti)   
                        self.savedf(name)
                    #print(pti,end=",") 
                    row=self.dfpt.iloc[pti].values
                    if np.any(row[ND:ND+2*ND]==1):#it can grow
                        #for each of the dimensions where it can grow, pull in that direction as long as it means it is getting away from center of mass
                        point=row[0:ND]
                        parset=row[ND+2*ND:]
                                
                        pt2projections=[]
                        for c,col in enumerate(self.colsgrow):
                            point2=point.copy()
                            #print("col", col)
                            dim,direction=col.split("_")
                            d=int(dim)
                            
                            #dpt=row[dim]
                            centroiddist=np.sum((point-centroid)**2)
                            if direction=='d':
                                projection=max(point[d]-L_project,self.min_arrays[d])
                            else:
                                projection=min(point[d]+L_project,self.maxbin_arrays[d])
                            point2[d]=projection
                            newcentroiddist=np.sum((point2-centroid)**2)
                            if newcentroiddist>centroiddist:
                                pt2projections.append(point2)
                            
                        for point2 in pt2projections:                                       
                            self.approach_target(niters_target,parset,point, point2, prob_replace=prob_replace,prob_par=prob_par,extr_uniform=extr_uniform,tolerance=tol_target)
                        
                        avprojection=np.sum(pt2projections)
                        self.approach_target(niters_target,parset,point, point2, prob_replace=prob_replace,prob_par=prob_par,extr_uniform=extr_uniform,tolerance=tol_target)


                                
        
                #newnboundary=len([x for x in self.dfpt[ND:ND+2*ND].values if np.any(x)==1])
                self.nboundary[i]=self.newpoints
                
            
            if verbose:
                print(i,self.nboundary[i],end='.,')
            if i>niters_conv:
                if np.sum(self.nboundary[i-niters_conv:i])<1:
                    self.converged=True
                
            i+=1
    
    
                
                    
            
        
        
        
        

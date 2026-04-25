"""
    Code to compute boundaries of a model in a 2D feature space.
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
import numpy as np
import matplotlib.pyplot as plt
import copy, os, sys
from scipy.optimize import fmin, fsolve
import itertools


def compute_pos_s_fsolve(pars,dfunc=None,func=None):
    """Computes position and steepness using negative of the derivative (dfunc) and the function minus 0.5 to find the normalisation xvalue"""
    
    ret=[None,None] #in case optimization fails, it will return this

    out=fmin(dfunc,1,args=(pars,),full_output=1,disp=False)

    if len(out)>0:
        xDER,fvalDER=out[0:2]
        xDER=xDER[0]
        fvalDER=-fvalDER
        
        out=fsolve(func,1,args=(pars,))
        if len(out)>0:

            x05=out[0]
            #Here one should make sure that x05 is positive and real!
        
            #rescale by x05
            fvalDERnorm=fvalDER*x05;
            xDERnorm=xDER/x05;
            ret=[xDERnorm,fvalDERnorm]  
    return ret

def compute_pos_s_mixed(pars,dfunc=None,x05f=None):
    """Computes position and steepness using negative of the derivative (dfunc) and the coefficients of the polynomial such that GRF=0.5 to find x05 using x05f"""
    
    ret=[None,None] #in case optimization fails, it will return this


    out=fmin(dfunc,1,args=(pars,),full_output=1,disp=False,ftol=0.00000005)

    if len(out)>0:
        xDER,fvalDER=out[0:2]
        xDER=xDER[0]
        fvalDER=-fvalDER

        if xDER > 0:

            x05=x05f(pars)
            if x05 is not None:
                if x05>0: #If it could not find x05 when x05f runs in c, then it returns -1
                    x05=np.real(x05) #this is necessary otherwise it has '0j' that triggers warning
            
                    #rescale by x05
                    fvalDERnorm=fvalDER*x05;
                    xDERnorm=xDER/x05;
                    ret=[xDERnorm,fvalDERnorm]  
    return ret





def position_steepness_hill(n,verbose=False):
    gammaHill=((n-1)/(n+1))**(1/n) #analytic position value
    rhoHill=((n**2-1)/(4.*n))*((n+1)/(n-1))**(1./n);
    
    return [gammaHill,rhoHill]



def p_s_indices(pos_ar,stp_ar,p_s):
    #given a value of p and s, returns their corresponding bin in the pos_ar and stp_ar arrays
    p,s=p_s
    #check if a given position (p) steepness (s) point it is within the preset bounds
    if s>=stp_ar[0] and s<stp_ar[-1]:
        idx_s=np.where(s>=stp_ar)[0][-1]
    else:
        idx_s=None
    if p>= pos_ar[0] and p<pos_ar[-1]:
        idx_p=np.where(p>=pos_ar)[0][-1]
    else:
        idx_p=None
    return [idx_p,idx_s]
def project_point(point,centroid,L=2,maxx=0,maxy=0):
    #given a boundary point and a centroid, gets the target point. Assumes that the boudary is more or less convex
    x0,y0=point
    xc,yc=centroid
    Dx=(x0-xc)
    Dy=y0-yc
    done=False
    while not done and L > 1:
        if Dx!=0:
            slope=Dy/Dx #can be positive, negative or 0
            if Dx>0:
                x=x0+np.sqrt(L*L/(1+slope**2))
                y=y0+slope*(x-x0)
            else:
                x=x0-np.sqrt(L*L/(1+slope**2))
                y=y0+slope*(x-x0)

        else:
            if Dy>0:
                x=x0
                y=y0+L

            else:
                x=x0
                y=y0-L
        if x>=0 and x<=maxx and y>=0 and y<=maxy:
            done=True
        else:
            L=L*0.9
    if done:
        return (x,y)
    else:
        return (None,None)



class BoundaryExplorer():
    def __init__(self,compute_x_y_f=None,pars_limit=[1e-5,1e5],constraints=None,npars=0,wpindices=[None,None],cg0=-1,nsites=0,row_ar=None,col_ar=None,seed=1,mat=None,mat_pars=None,tol_difference_pars=1e-8,ratios_indices=[None]):
        """This is a Class to found the boundaries of a model in 2D parameter space. For historic reasons it uses position/steepness for the variable notation and so on, but it  works for any 2D feature space. 
        By starting randomly from a few points, it will iteratively change them in various ways in order to find new points that expand the boundary.
        It uses a grid, and fills that grid. It outputs a matrix (mat) with 1 or 0 depending upon a point was found, and a matrix (mat_pars) with the corresponding parameter values. 

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
        - mat: in case one wants to continue a previous search, this should be the numpy 2D array with 1/0. 
        - mat_pars: in case of a continuation of a previous search, corresponding parameter values. 
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
        self.row_ar=row_ar #array defining the row bins of the grid
        self.col_ar=col_ar #array defining the column bins of the grid
        self.nrow=len(row_ar)
        self.ncol=len(col_ar)
        self.mchanges=None
        self.cg0=cg0
        if self.cg0>0:

            #check that none of the parameters greater or equal to this one is in hte constraint dictionary with target. Not implemented
            if constraints is not None:
                for key in constraints.keys():
                    if key>=self.cg0:
                        subdict=constraints[key]
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
                    #what is commented was used to ensure the weakest condition for monotonicity originally identified
                    #where w_p,s>w_p,s' for s'\in s
                    #now it uses #s>#s'
                    #if ns==2:
                    #    combis=sites
                    #else:
                    #    combis_=itertools.combinations(sites,ns-1)
                    #    combis=[]
                    #    for x in combis_:
                    #        combis.append(','.join(sorted(x)))
                    for j in range(0,i):
                        sitesprev=self.polstrlist[j].split('p')[1].split(",")
                        if len(sitesprev)==ns-1:
                        #if sitesprev in combis:
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


        self.constraints=constraint_mat
        
        self.compute_x_y=compute_x_y_f
        
        
        if mat is None: 
            self.mat=np.zeros((self.nrow,self.ncol))
        else:
            self.mat=mat
        if mat_pars is None:
            self.mat_pars=np.zeros((self.nrow,self.ncol,self.npars)) #parameter values 
        else:
            self.mat_pars=mat_pars
            
        #plt.imshow(self.mat)
        #plt.show()
        self.pos_val=np.zeros_like(self.mat)
        self.stp_val=np.zeros_like(self.mat)
        if np.sum(self.mat)>0:
            for row in range(self.nrow):
                for col in range(self.ncol):
                    if self.mat[row,col]>0:
                        self.pos_val[row,col]=self.col_ar[col]
                        self.stp_val[row,col]=self.row_ar[row]
        
            
        
        self.target=False
        self.pulltangents=False
    
        np.random.seed(seed)
        
        self.indices_boundary_expl=None
        self.indices_boundary_all=None
        
        self.nboundary=None #how many new points in the boundary at each iteration of extend_boundary
    
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
        
    def get_initial_points(self,ninitpoints):
        #Generate ninitpoints to begin with
        i=0
        
        while i < ninitpoints:
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

            #print("testing pars")
            #print(",".join(map(str,pars)))
            #p=None
            p,s=self.compute_x_y(pars)
            #p,s=compute_pos_s_inputoutput(self.xrange,pars,self.ssfunc)
            if p is not None:
                idx_p,idx_s=p_s_indices(self.col_ar,self.row_ar,[p,s])
                if idx_p is not None and idx_s is not None:
                    if self.mat[idx_s,idx_p]<1:
                        self.mat[idx_s,idx_p]=1
                        self.mat_pars[idx_s,idx_p]=pars
                        self.pos_val[idx_s,idx_p]=p
                        self.stp_val[idx_s,idx_p]=s
                        i+=1
                        
    def find_boundary_points(self):
        """
        fills 2 matrices:
        indices_boundary_expl: with True on the positions where there is a boundary point that will be taken to explore (False for instance where it hits the limits of the predefined grid array.)
        indices_boundary_all: with True on all boundary positions
        """
        mat=self.mat
        self.indices_boundary_expl=np.zeros_like(mat,dtype=bool)
        self.indices_boundary_all=np.zeros_like(mat,dtype=bool)
        
        for row in range(self.nrow):
            non0=np.where(mat[row]>0)[0]#extremes of the row
            if len(non0)>0:
                if len(non0)==1:
                    i0=non0[0]
                    i1=None
                else:
                    i0,i1=non0[np.array([0,-1])] 
                for idx in [i0,i1]:
                    if idx is not None:
                        self.indices_boundary_all[row,idx]=1
                        #check if it is in the preset boundary
                        nonfree=[0,0]
                        if idx==0 or idx==(self.ncol-1):#first or last column
                            row_up=row-1
                            row_down=row+1
                            if row_up>0:
                                if mat[row_up][idx]>0:
                                    nonfree[0]=1
                            else:
                                nonfree[0]=1
                            if row_down<self.nrow:
                                if mat[row_down][idx]>0:
                                    nonfree[1]=1
                            else:
                                nonfree[1]=1
                        if sum(nonfree)<2:
                            self.indices_boundary_expl[row,idx]=1

        for col in range(self.ncol):
            non0=np.where(mat[:,col]>0)[0]#extremes of the col
            if len(non0)>0:
                if len(non0)==1:
                    i0=non0[0]
                    i1=None
                else:
                    i0,i1=non0[np.array([0,-1])] 

                for idx in [i0,i1]:
                    if idx is not None:
                        self.indices_boundary_all[idx,col]=1
                        #check if it is in the preset boundary
                        nonfree=[0,0]
                        if idx==0 or idx==(self.nrow-1):#first or last row
                            col_l=col-1
                            col_r=col+1
                            if col_l>0:
                                if mat[idx][col_l]>0:
                                    nonfree[0]=1
                            else:
                                nonfree[0]=1
                            if col_r<self.ncol:
                                if mat[idx][col_r]>0:
                                    nonfree[1]=1
                            else:
                                nonfree[1]=1
                           
                        if sum(nonfree)<2:
                            self.indices_boundary_expl[idx,col]=1
        
    def find_delimiting(self,idx_p,idx_s,mat,tol=10**(-5)):
        #finds if a given p,s is at the limit of any of the position/steepness arrays or is at the boundary (not including this point)
        if idx_p==self.ncol-1:
            c_r=True #limits on the right
            c_l=(np.sum(mat[idx_s,0:idx_p])<tol)#True if there everything to the left is 0, so it limits on the left
        elif idx_p==0:
            c_l=True #limits on the left
            c_r=(np.sum(mat[idx_s,idx_p+1:])<tol) #limits on the right
        else:
            c_r=(np.sum(mat[idx_s,idx_p+1:])<tol) 
            c_l=(np.sum(mat[idx_s,0:idx_p])<tol) 

        if idx_s==self.nrow-1:
            c_d=True #limits down
            c_u=(np.sum(mat[0:idx_s,idx_p])<tol) #limits up
        elif idx_s==0:
            c_u=True
            c_d=(np.sum(mat[idx_s+1:,idx_p])<tol) #limits down
        else:
            c_u=(np.sum(mat[0:idx_s,idx_p])<tol)
            c_d=(np.sum(mat[idx_s+1:,idx_p])<tol)
        return [c_l,c_r,c_u,c_d]
    
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
                    
                    if np.abs(min_-max_)<0.0001:#equal:
                        newp=10**min_
                    else:
                        min10=10**min_
                        max10=10**max_

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
                                if lt<0 and gt<0:
                                    newp=np.random.normal(targetvalue,targetvalue/factor)
                                else:
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
        
                p,s=self.compute_x_y(pars)
                #p,s=compute_pos_s_inputoutput(self.xrange,pars,self.ssfunc)

                if p is not None:
                    idx_p,idx_s=p_s_indices(self.col_ar,self.row_ar,[p,s])


                    if (idx_p is not None) and (idx_s is not None):
                        #print('p=%g,idx_p=%d,s=%g,idx_s=%d'%(p,idx_p,s,idx_s))
                        #if True:
                        if targetted is True:
                            for_targetted=True
                        else:
                        
                            limits=self.find_delimiting(idx_p,idx_s,self.mat) #find if the new point becomes at the boundary
                            if np.any(limits):
                                flag=False
                                if self.mat[idx_s,idx_p]<1: #it was an empty spot
                                    flag=True
                                elif np.random.uniform(0,1) < prob_replace: #with probability prob_replace, allow replacing the boundary point
                                    flag=True

                                if flag is True:
                                    self.mat[idx_s,idx_p]=1
                                    self.mat_pars[idx_s,idx_p]=pars
                                    self.pos_val[idx_s,idx_p]=p
                                    self.stp_val[idx_s,idx_p]=s
                                    done=True    


                                
            
        if targetted:
            if for_targetted:
                return [p,s,pars]
            else:
                return [None,None,None]
        

            
    def approach_target(self,niters_target,parset,p0s0,tpts,extr_uniform=None,prob_par=1,prob_replace=0,tolerance=0.01):
        #will try to approach a given target position and steepness
        tp,ts=tpts #target position and steepness
        p0,s0=p0s0 #current position and steepness
        d0=(tp-p0)**2+(ts-s0)**2 #initial distance
        
        
        distances=np.zeros(niters_target)
        distances[0]=d0
       
        bestp=p0
        bests=s0
        
        i=1
        converged=False
        while i < niters_target and converged == False:
            d=-1
           
            p,s,pars=self.mutate_parset(parset,trials=1,prob_par=prob_par,prob_replace=prob_replace,extr_uniform=extr_uniform,targetted=True)
             
            if p is not None:
                d=(tp-p)**2+(ts-s)**2 
                 

            if d>0 and d<distances[i-1]:
                parset=pars #new pars are these ones
                bestp=p
                bests=s
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
        idx_p,idx_s=p_s_indices(self.col_ar,self.row_ar,[bestp,bests])


        if (idx_p is not None) and (idx_s is not None):
            #print('p=%g,idx_p=%d,s=%g,idx_s=%d'%(p,idx_p,s,idx_s))
            if self.mat[idx_s,idx_p]<1: #it was an empty spot
                limits=self.find_delimiting(idx_p,idx_s,self.mat) #find if the new point becomes at the boundary
                if np.any(limits):
                    #print('saving improved point',idx_p,idx_s)
                    self.mat[idx_s,idx_p]=1
                    self.mat_pars[idx_s,idx_p]=parset
                    self.pos_val[idx_s,idx_p]=bestp
                    self.stp_val[idx_s,idx_p]=bests

                    #tocheck. I am commenting it as of 30 july
                    #p_2,s_2=self.compute_x_y(parset)
                    #idx_p_2,idx_s_2=p_s_indices(self.col_ar,self.row_ar,[p_2,s_2])
                    #if idx_p!= idx_p_2 or idx_s!=idx_s_2:
                    #    print('*difference here in approach target!!!')
                        
    def project_point_2(self,ptnum,pt,currentpts,L=None,w=None,theta=None): 
        be=self.indices_boundary_expl
        nrow=self.nrow
        ncol=self.ncol
        #indexes of the projection point
        idx_p=None
        idx_s=None
        
        
        x1,y1=pt
        l=x1
        l_=min(x1+w,ncol-1)
        foundn=False
        m2=0 #slope of projection line
        
        while l<l_ and foundn is False: #attempt to find closest to the right at most w away. There may be nothing to the right.
            l+=1
            extremes2=np.where(be[:,l]>0)[0] #boundary rows for this column. Can be 1 or two, depending on whether grid extremes have already been reached.
            if len(extremes2)>0: #if at least there is one point to the right

                difrow=np.where(np.abs(extremes2-y1)<w)[0]
                if len(difrow)>0: #there can be more than 1 neighbor if one or more are row border and the other is column border. I will pick the first one
                    if np.all(np.abs(difrow) < w): #otherwise, it could be taking the one corresponding to the other part of the boundary
                        foundn=True
        
        if foundn is True:
                        
            y2=extremes2[difrow[0]] #even if there is more than 1 neighbor, I just pick one
            x2=l

            pt2=[x2,y2]
            m1=(y2-y1)/(x2-x1) #slope of tangent line to point pt

            if np.abs(m1)<0.00001: #basically 0 slope
                m2=np.inf
            else:
                vertical=False
                if m1>0:

                    #m=-1/m #perpendicular

                    alpha=np.arctan(m1)
                    alpha_theta=alpha+theta
                    if np.abs(alpha_theta-np.pi/2)<0.0001:
                        #vertical line
                        m2=np.inf
                    else:
                        m2=np.tan(alpha_theta)
                else:
                    m2=np.tan(theta)

        if np.isinf(m2):#vertical projection line
            #try up or down
            #try up:
            up=False
            done=False
            xu=x1
            L_=L
            
            while not done and L_>1:
                yu=int(y1+L_)
                if yu<nrow:
                    done=True
                else:
                    L_=L_*0.9
            
            if done:
                idx_pu=int(xu)
                idx_su=int(yu)
                limits=self.find_delimiting(idx_pu,idx_su,self.mat) 
                if np.any(limits):
                    if self.mat[idx_su,idx_pu]<1: #it was an empty spot
                        up=True

            #try down
            down=False
            done=False
            xd=x1
            L_=L
            
            while not done and L_>1:
                yd=int(y1-L_)
                if yd>0:
                    done=True
                else:
                    L_=L_*0.9
                    
            if done:
                idx_pd=int(xd)
                idx_sd=int(yd)
                limits=self.find_delimiting(idx_pd,idx_sd,self.mat)
                if np.any(limits):
                    if self.mat[idx_sd,idx_pd]<1: #it was an empty spot
                        down=True

            if up and down:
                #check which one of the two depedning on whether there are points in between. 

                yvals=np.arange(y1+1,yu)
                xc=x1
                for k in range(len(yvals)):
                    yc=int(yvals[k])
                    if self.mat[yc,xc]>0:
                        up=False

                yvals=np.arange(yd,y1-1)
                for k in range(len(yvals)):
                    yc=int(yvals[k])
                    if self.mat[yc,xc]>0:
                        down=False

            if up:
                #plt.scatter(pos_ar[idx_pr],stp_ar[idx_sr],color='r')
                idx_p=idx_pu
                idx_s=idx_su
            if down:
                idx_p=idx_pd
                idx_s=idx_sd
        else: #either horizontal or certain angle projection line       
            horizontal=False
            if foundn is False:                 
                if len(currentpts)>1: #at least there is one other point on the same column
                    #now check if there is a point 
                    if ptnum+1 < len(currentpts):
                        pt2=currentpts[ptnum+1]
                        y2=pt2[1]
                        if y2-y1<w: #if the following point is vertically adjacent
                            horizontal=True
            
            if np.abs(m2)>0 or horizontal:
                
                #Try either right or left

                x0=x1
                y0=y1
                #try right:
                right=False
                
                done=False
                L_=L
                if horizontal:
                    yr=y0
                    while not done and L_>1:
                        xr=int(x0+L_)
                        if xr>=ncol:
                            L_=L_*0.9
                        else:
                            done=True
                else:
                    while not done and L_>1:
                        slope=m2
                        xr=int(x0+np.sqrt(L_*L_/(1+slope**2)))
                        yr=int(y0+slope*(xr-x0))
                        if xr<ncol and yr<nrow and yr>0:
                            done=True
                        else:
                            L_=L_*0.9
                    
                if done: 
                    idx_pr=int(xr)
                    idx_sr=int(yr)
                    limits=self.find_delimiting(idx_pr,idx_sr,self.mat)
                    if np.any(limits):
                        if self.mat[idx_sr,idx_pr]<1: #it was an empty spot
                            right=True


                #try left
                left=False
                done=False
                if horizontal:
                    yl=y0
                    xl=x0-L_
                    while not done and L_>1:
                        xl=int(x0-L_)
                        if xl<0:
                            L_=L_*0.9
                        else:
                            done=True
                    
                else:
                    while not done and L_>1:
                        slope=m2
                        xl=x0-np.sqrt(L_*L_/(1+slope**2))
                        yl=y0+slope*(xl-x0)
                        if xl>0 and yl<nrow and yl>0:
                            done=True
                        else:
                            L_=L_*0.9
                if done:    
                    idx_pl=int(xl)
                    idx_sl=int(yl)
                    limits=self.find_delimiting(idx_pl,idx_sl,self.mat)
                    if np.any(limits):
                        if self.mat[idx_sl,idx_pl]<1: #it was an empty spot
                            left=True

                if right and left:
                    #check which one of the two depedning on whether there are points in between. 
                    xvals=np.arange(x0+1,xr,1)
                    b2=y1-m2*x1
                    yvals=m2*xvals+b2
                    for k in range(len(xvals)):
                        xc=int(xvals[k])
                        yc=int(yvals[k])
                        if self.mat[yc,xc]>0:
                            right=False


                    xvals=np.arange(xl,x0-1,1)
                    yvals=m2*xvals+b2
                    for k in range(len(xvals)):
                        xc=int(xvals[k])
                        yc=int(yvals[k])
                        if self.mat[yc,xc]>0:
                            left=False

                if right:
                    #plt.scatter(pos_ar[idx_pr],stp_ar[idx_sr],color='r')
                    idx_p=idx_pr
                    idx_s=idx_sr
                if left:
                    idx_p=idx_pl
                    idx_s=idx_sl
                    #plt.scatter(pos_ar[idx_pl],stp_ar[idx_sl],color='cyan')




        if idx_p and idx_s:
            return [idx_p,idx_s]
        else:
            return None



    def plot_boundary(self,title=None,show=True):
        extent=[self.col_ar[0],self.col_ar[-1],self.row_ar[0],self.row_ar[-1]]
        fig,ax=plt.subplots(1,1,figsize=(6,5))
        ax.imshow(self.mat,origin='lower',extent=extent,cmap=plt.cm.Greys)
        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.indices_boundary_expl[row,col]>0:
                    ax.scatter(self.col_ar[col],self.row_ar[row],color='g',s=5)
        ax.set_title(title)
        if show:
            plt.show()
        return [fig,ax]


    def extend_boundary(self,dofirstmutate=True,radius=1, maxradius=10, dopulltangents=False, theta=np.pi/4, dopullcentroids=True,
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
        
        self.converged=False
        prev_mat=self.mat.copy()
        mat_changes=np.zeros_like(self.mat)
        i=0
        self.nboundary=np.zeros(niters)
        self.extr_uniform=extr_uniform
        self.radius=radius
        self.eraseintermediates=eraseintermediates
        while i < niters and self.converged== False:
            if i%niters_save ==0:
                name='%s_%i'%(name_save,i)
                matname=os.path.join(folder_save,'mat_'+name+'.npy')
                matparsname=os.path.join(folder_save,'mat_pars_'+name+'.npy')
                np.save(matname,self.mat)
                np.save(matparsname,self.mat_pars)
                if self.eraseintermediates is True:
                	if i>0:
                		#delete previous
                		os.remove(prevmatname)
                		os.remove(prevmatparsname)
                	prevmatname=matname
                	prevmatparsname=matparsname


            print(i,end=",",flush=True)
            #this could be made to depend upon the iteration
            fmin=self.extr_uniform[0]
            fmax=self.extr_uniform[1]
            
            if dofirstmutate:
                print("Doing firstmutate")

                if i > 2:
                    if np.sum(self.nboundary[i-2:i])<1: #when it seems that it starts stopping, try more mutations per parameter
                        if self.radius+1 <= maxradius:
                            self.radius=self.radius+1
                
                point_coords_expl=np.where(self.indices_boundary_expl>0)
                for j in range(len(point_coords_expl[0])): #for each point in the boundary
                    print("%d"%j,end=",")
                    sys.stdout.flush()
                    y=point_coords_expl[0][j]
                    x=point_coords_expl[1][j]
                    if mat_changes[y,x]<niters_conv_points:
                        parset=self.mat_pars[y,x]
                        for j in range(self.radius):#radius number of times 
                            self.mutate_parset(parset,prob_par=prob_par,prob_replace=prob_replace,extr_uniform=extr_uniform)
                
                        
                self.find_boundary_points()
                self.nboundary[i]=np.sum(self.mat-prev_mat)
                if plotting:
                    self.plot_boundary('first m,i=%d'%i)
            
            if dopullcentroids and self.target is False:
                if np.sum(self.indices_boundary_all)>10:
                    self.target=True
            
            if dopulltangents and self.pulltangents is False:
                if np.sum(self.indices_boundary_expl)>100:  #only do it if there are sufficient boundary points
                    self.pulltangents=True
            print("Doing pulltangents")
            if self.pulltangents:
                w=3 #vertical distance between a point and the one considered neighboring in order to compute tangent. If too large, it could take a point on the other side of the boundary.  
                #theta=np.pi/4 #angle for pulling. This could be modified or set as a parameter passed as argument for the extend_boundary function
                be=self.indices_boundary_expl
                
                for j in np.arange(1,self.ncol-2):
                    extremes=np.where(be[:,j]>0) ##boundary rows for this column. Can be 1,2 or even more if they are row boundaries as well, and depending on whether grid extremes have already been reached.
                    if len(extremes[0])>0: #it could be that no points for this column
                        currentpts=[[j,row] for row in extremes[0]] #position idx, steepness idx for each of the boundary points
                        #print("at col ",j,currentpts)
                        
                        for ptnum,point in enumerate(currentpts): #for each 
                            print("point %d"%ptnum,end=",")
                            sys.stdout.flush()
                            
                            
                            projection=self.project_point_2(ptnum,point,currentpts,L=L_project,w=w,theta=theta)
                            if projection is not None:
                                idx_p,idx_s=projection
                                

                                #print(xp,yp,self.nrow,self.ncol)
                                #idx_p=int(xp) #np.where(xp>=self.col_ar)[0][-1]
                                #idx_s=int(yp) #np.where(yp>=self.row_ar)[0][-1]
                                #limits=self.find_delimiting(idx_p,idx_s,self.mat) #check that this new target point would in fact be at the boundary
                                #if np.any(limits) and self.mat[idx_s,idx_p]<1:
                                x,y=point
                                p=self.pos_val[y,x] #current position
                                s=self.stp_val[y,x] #current steepness
                                pt=self.col_ar[idx_p] #target position
                                st=self.row_ar[idx_s] #target steepness
                                if False:
                                    boundaryidxs=np.where(be>0)
                                    plt.scatter(boundaryidxs[1],boundaryidxs[0],color='grey')
                                    plt.scatter(x,y,color='k')
                                    #plt.scatter(idx_p,idx_s,color='orange')
                                    plt.plot([x,idx_p],[y,idx_s],color='r')
                                    print('starting at ',p,s,'with indices',idx_p,idx_s,'trying to go to',pt,st)
                                
                                self.approach_target(niters_target,self.mat_pars[y,x].copy(),[p,s],[pt,st],prob_replace=prob_replace,prob_par=prob_par,extr_uniform=extr_uniform,tolerance=tol_target)

                self.find_boundary_points()
                self.nboundary[i]=np.sum(self.mat-prev_mat)

                if plotting:
                    #plt.show()
                    self.plot_boundary('after pulling tangents, i=%d'%i)
            if self.target:
                print("Doing target")

            #if i > 2:
                #if True:
                #if np.sum(self.nboundary[i-2:i])<1:#if it didn't improve for 2 rounds, try to approach a target
                    #print('optimizing')
                point_coords_all=np.where(self.indices_boundary_all>0)
                point_coords_expl=np.where(self.indices_boundary_expl>0)
                yc=np.mean(point_coords_all[0]) #y coord of centroid (steepness)
                xc=np.mean(point_coords_all[1]) #x coord of centroid (position)
                if plotting:
                    plt.imshow(self.indices_boundary_expl,origin='lower',cmap=plt.cm.Greys)
                    plt.scatter(xc,yc,color='r')
                    
                for j in range(len(point_coords_expl[0])):
                    print("%d"%j,end=",")
                    sys.stdout.flush()
                    y=point_coords_expl[0][j]
                    x=point_coords_expl[1][j]
                    if mat_changes[y,x]<niters_conv_points: #if the point didn't change for niters_conv iterations, do not try any more. This is to speed up. 

                        #print(x,y)
                        #print(indices_boundary_explore[y,x])
                        #print(mat_pars[y,x])
                        xp,yp=project_point([x,y],[xc,yc],L=L_project,maxx=self.ncol-1,maxy=self.nrow-1) #find the target point corresponding to this boundary point

                        if xp is not None:

                            if plotting:
                                plt.scatter(x,y,color='b')
                                plt.scatter(xp,yp,color='orange')
                                plt.plot([x,xp],[y,yp],color='r')

                            #print(xp,yp,self.nrow,self.ncol)
                            idx_p=int(xp) #np.where(xp>=self.col_ar)[0][-1]
                            idx_s=int(yp) #np.where(yp>=self.row_ar)[0][-1]
                            limits=self.find_delimiting(idx_p,idx_s,self.mat) #check that this new target point would in fact be at the boundary
                            if np.any(limits) and self.mat[idx_s,idx_p]<1:
                                p=self.pos_val[y,x] #current position
                                s=self.stp_val[y,x] #current steepness
                                pt=self.col_ar[idx_p] #target position
                                st=self.row_ar[idx_s] #target steepness
                                
                                self.approach_target(niters_target,self.mat_pars[y,x].copy(),[p,s],[pt,st],prob_replace=prob_replace,prob_par=prob_par,extr_uniform=extr_uniform,tolerance=tol_target)
                        
                                
        
                self.find_boundary_points()
                self.nboundary[i]=np.sum(self.mat-prev_mat)
                




                if plotting:
                    plt.show()
                    self.plot_boundary('after pulling centroids, i=%d'%i)

            mdif=(self.mat-prev_mat)
            mat_changes[self.indices_boundary_all]=mat_changes[self.indices_boundary_all]-(mdif[self.indices_boundary_all]-1)  #mdif-1: newly occupied positions: 0, previously occupied position:-1. By adding the negative of this, you are increasing by one the count of those boundary positions that did not change
            
            prev_mat=self.mat.copy()
            if verbose:
                print(i,self.nboundary[i],end='.,')
            if i>niters_conv:
                if np.sum(self.nboundary[i-niters_conv:i])<1:
                    self.converged=True
                    self.mchanges=mat_changes
                
            i+=1
    
    
                
                    
            
        
        
        
        

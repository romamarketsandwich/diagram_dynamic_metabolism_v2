import numpy as np
import generatehypercube

#Start by generating a hypercube graph with N+1 sites


def find_reverse(n1,n2,edges):
    for enumr,er in enumerate(edges):
        n1_,label,n2_=er
        if n1_==n2 and n2_==n1:
            return [enumr,er]

def get_ratios_andreferences_list(edges): 
    result=[]
    for enum,e in enumerate(edges):
        if '-P' in e[1]:
            if not 'e' in e[1]:
                out=[[],[],0]
                n1,label,n2=e
                #print("\n.....e",e)
                #find reverse edge
                eridx,er=find_reverse(n1,n2,edges)
                
                out[0].append(enum)
                out[0].append(eridx)
                
                #print("reverse is", er)
                #find edge with Pol binding and the other sites occupied except one
                sites=set(list(e[1].split('-')[0])[2:])
                nsites=len(sites)
                out[2]=nsites
                if nsites==1:
                    #compare to empty
                    for enum2,e2 in enumerate(edges):
                        lab=e2[1]
                        if '-P' in lab and 'e' in lab:
                            #print("empty", e2)
                            eridx,er=find_reverse(e2[0],e2[2],edges)
                            out[1].append([enum2,eridx])

                else:
                    for enum2,e2 in enumerate(edges):
                        lab=e2[1]
                        if '-P' in lab and not 'e' in lab:
                            sites2=set(list(lab.split('-')[0])[2:])
                            if len(sites2)==len(sites)-1:
                                if len(sites2.difference(sites))==0:
                                    #print("contains all sites", e2)
                                    erevidx2,er2=find_reverse(e2[0],e2[2],edges)
                                    #print("reverse is", er2)
                                    out[1].append([enum2,erevidx2])
                #print(edges[out[0][0]],edges[out[0][1]])
                #print("ratio compared to:")
                #for out_ in out[1]:
                #    print(out_)
                #    print(edges[out_[0]],edges[out_[1]])
                result.append(out)

    return sorted(result,key=lambda x:x[2])

def get_ratios_indices(N):
    """N is the number of binding sites, excluding Pol"""

    G=generatehypercube.hypercube(N+1) 

    edges_=[]
    for edge in G.edges:
        #print(edge,G2[edge[0]][edge[1]]['weight'])
        tuple_=(edge[0],G[edge[0]][edge[1]]['k'],edge[1])
        edges_.append(tuple_)

    #Now catch those binding edges to the Pol site

    edges=[]
    nodeswithP=set()
    for edge in edges_:
        n0,label,n1=edge
        site0=label[1]
        if label[0]=="a" and site0==str(N+1):
            #print(n0,n1,label)
            label=label.replace("x","P")
            nodeswithP.add(n1)
            
        edges.append((n0,label,n1))

    return get_ratios_andreferences_list(edges)

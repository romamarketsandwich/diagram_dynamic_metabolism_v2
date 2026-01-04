import networkx as nx

def get_nextbinding(bsites,allsites,free_sites,G):
    #produces graph with 'a' and 'b' as labels
    for s in free_sites:
        currentsites=''.join(map(str,sorted(bsites)))
        microstate1='i%s'%currentsites
        bsitesn=bsites+[s] #bound sites after binding to s
        if bsitesn[0]=='e':
            bsitesn.pop(0) #it is not longer empty
        nextsites=''.join(map(str,sorted(bsitesn)))
        microstate2='i%s'%(nextsites)
        konstr='a%d%s-x'%(s,currentsites)
        koffstr='b%d%s'%(s,nextsites)
        #konstr='a%d%s'%(s,currentsites)
        #k
        G.add_edge(microstate1,microstate2,k=konstr)
        G.add_edge(microstate2,microstate1,k=koffstr)
        
        #G.add_edge(microstate2,microstate1,k=koffstr)
        free_sitesn=[x for x in allsites if not x in bsitesn]
        if len(free_sites)>0:
            get_nextbinding(bsitesn,allsites,free_sitesn,G)

def hypercube(N):

	G=nx.DiGraph()
	get_nextbinding(['e'],range(1,N+1),range(1,N+1),G)
	node_dict={}
	for n, node in enumerate(nx.traversal.bfs_tree(G,'ie')):
	    node_dict[node]=n+1
	    #print(node,n+1)
	G2=nx.relabel_nodes(G,node_dict)
	return G2
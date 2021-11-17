import pandas as pd
import numpy as np


path='/home/mengxuan/Desktop/GE2324/project/data/indexed_location.csv'
nodes = pd.read_csv(path,header=0,index_col=0)

dic={
    'central':1,
    'hongkong':2,
    'kowloon':3,
    'lantau':4,
    'northeast':5,
    'northwest':6,
    'outlying':7,
    'saikung':8
}
dic2={
    1:'central',
    2:'hongkong',
    3:'kowloon',
    4:'lantau',
    5:'northeast',
    6:'northwest',
    7:'outlying',
    8:'saikung'
}
df1 = df.replace({'loc': dic})

for i in range(1,9):
    for j in range(i+1,9):
        df_1 = df1[df1['loc']==i]
        df_2 = df1[df1['loc']==j]
        print 'intersection between',dic2[i],'and',dic2[j],'is',str(set(df_1.index).intersection(df_2.index))

# df1.to_csv('/home/mengxuan/Desktop/GE2324/project/data/indexed_location.csv')

tmp = 0
for r in edges.values:
    try:
        if nodes.loc[r[0]]['loc'] != nodes.loc[r[1]]['loc']:
            tmp += r[3]
    except:
        print r

import pandas as pd
import numpy as np
path='/home/mengxuan/Desktop/GE2324/project/data'
pathre = '/home/mengxuan/Desktop/GE2324/project/result'
edges = pd.read_csv(path+'/edge_table.csv',header=0)
nodes = pd.read_csv(path+'/node_table_new.csv',header=0)
trail_nodeid = pd.read_csv(path+'/trail_nodeId_table.csv',header=0)


import datetime
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import operator
def centralization(g):
    N=g.order()
    degrees = g.degree().values()
    max_deg = max(degrees)
    if N-2 <= 0:
        res = -1
    else:
        res = float((N*max_deg - sum(degrees)))/((N-1)*(N-2))
    return res
def drawNet():
    max_clo={}
    max_bet={}
    networks={}
    ed = zip(edges['source'],edges['target'])
    gr = nx.Graph()
    gr.add_nodes_from(nodes['id'])
    gr.add_edges_from(ed)
    usedG = []
    res = pd.DataFrame(columns=['subnetworkid','betweenness','closeness','centralization'])
    for i in gr.nodes():
        s = gr.subgraph(nx.shortest_path(gr.to_undirected(),i))
        if s.nodes()[0] not in usedG:
            print str(datetime.datetime.now()),'drawNet: drawing for:',str(i),'network'
            usedG+=s.nodes()
            networks[i]=s.nodes()
            ns = s.nodes()
            plt.clf()
            pos = nx.spring_layout(s,k=0.15,iterations=20)
            nx.draw(s, pos, node_size=10, with_labels=True)
            plt.savefig(pathre+'/network_'+str(i)+'.png')
            plt.clf()
            ind = np.arange(len(nx.betweenness_centrality(s)))
            fig, ax = plt.subplots(figsize=(20,10))
            fig.patch.set_facecolor('#333333')
            ax2 = ax.twinx()
            bar_width = 0.15
            opacity = 0.5
            bet = ax.bar(ind - bar_width/2, [nx.betweenness_centrality(s)[k] for k in ns], bar_width,
                             alpha=opacity,
                             color='#00c6ff',
                             label='betweeness')
            clo = ax2.bar(ind + bar_width/2, [nx.closeness_centrality(s)[k] for k in ns], bar_width,
                             alpha=opacity,
                             color='orange',
                             label='closeness')
            plt.xlabel('Nodes',color='white')
            ax.set_ylabel('Betweenness',color='white')
            ax2.set_ylabel('Closeness',color='white')
            plt.title('Network Centrality '+str(i),color='white')
            ax.legend(loc=2)
            ax2.legend(loc=1)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax2.spines['bottom'].set_color('white')
            ax2.spines['top'].set_color('white') 
            ax2.spines['right'].set_color('white')
            ax2.spines['left'].set_color('white')
            for t in ax.yaxis.get_ticklabels(): t.set_color('white')
            for t in ax2.yaxis.get_ticklabels(): t.set_color('white')
            l = []
            for k in ns:
                try:
                    l.append(str(k)+' '+nodes[nodes.id==k]['spot_en'].values[0])
                except:
                    l.append(k)
                    print k
            ax.set_xticks(ind)
            ax.set_xticklabels(l,color='white',rotation='vertical')
            plt.tight_layout()
            plt.savefig(pathre+'/network_'+str(i)+'_centrality.png',transparent=True,facecolor=fig.get_facecolor(), edgecolor='none')
            print str(datetime.datetime.now()),'drawNet: finish for:',str(i),'network'
            res.loc[res.shape[0]] = [i,nx.betweenness_centrality(s),nx.closeness_centrality(s),centralization(s)]
            max_bet[i] = [sorted(nx.betweenness_centrality(s).items(), key=operator.itemgetter(1))[-1][0],sorted(nx.betweenness_centrality(s).items(), key=operator.itemgetter(1))[-1][1]]
            max_clo[i] = [sorted(nx.closeness_centrality(s).items(), key=operator.itemgetter(1))[-1][0],sorted(nx.closeness_centrality(s).items(), key=operator.itemgetter(1))[-1][1]]
    df_bet = pd.DataFrame(index=networks.keys(),columns=['betweenness','node_id','spot_en','spot_ch','loc'])
    df_clo = pd.DataFrame(index=networks.keys(),columns=['closeness','node_id','spot_en','spot_ch','loc'])
    for a in networks.keys():
        try:
            df_bet.ix[a,'betweenness'] = max_bet[a][1]
            df_bet.ix[a,'node_id'] = max_bet[a][0]
            df_bet.ix[a,'spot_en'] = nodes[nodes.id==max_bet[a][0]]['spot_en'].values[0]
            df_bet.ix[a,'spot_ch'] = nodes[nodes.id==max_bet[a][0]]['spot_ch'].values[0]
            df_bet.ix[a,'loc'] = nodes[nodes.id==max_bet[a][0]]['loc'].values[0]
            df_clo.ix[a,'closeness'] = max_clo[a][1]
            df_clo.ix[a,'node_id'] = max_clo[a][0]
            df_clo.ix[a,'spot_en'] = nodes[nodes.id==max_clo[a][0]]['spot_en'].values[0]
            df_clo.ix[a,'spot_ch'] = nodes[nodes.id==max_clo[a][0]]['spot_ch'].values[0]
            df_clo.ix[a,'loc'] = nodes[nodes.id==max_clo[a][0]]['loc'].values[0]
        except:
            print a
    df_bet.to_csv(pathre+'/betweenness_statistics.csv')
    df_clo.to_csv(pathre+'/closeness_statistics.csv')
    res.to_csv(pathre+'/centrality.csv')
    return networks

def findNets():
    max_clo={}
    max_bet={}
    networks={}
    ed = zip(edges['source'],edges['target'])
    gr = nx.Graph()
    gr.add_nodes_from(nodes['id'])
    gr.add_edges_from(ed)
    usedG = []
    res = pd.DataFrame(columns=['subnetworkid','betweenness','closeness','centralization'])
    for i in gr.nodes():
        s = gr.subgraph(nx.shortest_path(gr.to_undirected(),i))
        if s.nodes()[0] not in usedG:
            usedG+=s.nodes()
            networks[i]=s.nodes()
    return networks

def whichNet(nodeid):
    for k in nets.keys():
        if nodeid in nets[k]:
            print 'network:',str(k)







base = range(3) #arbitrarily choose to start from nodes 0, 1, and 2
depth = 3  #look for those within length 3.
foundset = {key for source in base for key in nx.single_source_shortest_path(gr,source,cutoff=depth).keys()}
H=gr.subgraph(foundset)
plt.clf()
nx.draw_networkx(H)
plt.savefig(pathre+'/network.png') 

        

# closeness
gr = nx.Graph()
gr.add_nodes_from(range(1,13))
gr.add_edge(1,2)
gr.add_edge(1,3)
gr.add_edge(1,4)
gr.add_edge(1,5)
gr.add_edge(1,6)
gr.add_edge(1,7)
gr.add_edge(1,8)
gr.add_edge(1,9)
gr.add_edge(8,9)
gr.add_edge(8,10)
gr.add_edge(9,10)
gr.add_edge(10,12)
gr.add_edge(11,10)
gr.add_edge(11,12)
plt.clf()
mylabels = dict(zip(range(1,13),range(1,13)))
nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
plt.savefig(pathre+'/network.png')
nx.closeness_centrality(gr)

# betweenness
gr = nx.Graph()
gr.add_nodes_from(range(1,13))
gr.add_edge(1,2)
gr.add_edge(1,3)
gr.add_edge(2,4)
gr.add_edge(3,4)
gr.add_edge(4,5)
gr.add_edge(4,6)
gr.add_edge(4,7)
gr.add_edge(4,8)
gr.add_edge(4,10)
gr.add_edge(8,10)
gr.add_edge(7,9)
gr.add_edge(7,10)
gr.add_edge(10,11)
gr.add_edge(10,12)
plt.clf()
mylabels = dict(zip(range(1,13),range(1,13)))
nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
plt.savefig(pathre+'/network2.png')
nx.betweenness_centrality(gr)

# centralization


drawNet()

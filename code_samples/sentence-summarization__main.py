#Non-ML implementation
import matplotlib.pyplot as plt
import math
import string
import networkx as nx
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import requests

url = ('https://newsapi.org/v2/everything?'
       'q=Trump&'
       'sortBy=popularity&'
       'apiKey=2472ff0d28ac4fdbb250def86ab797f3')

response = requests.get(url)

#print(response.json())

headline_dict = response.json()
headline_titles = [dic['title'] for dic in headline_dict['articles']]
#process given test data and stopwords
"""testdata = open("bolton_majoritygarbage_mix.txt")"""
#stopwords = open("stopwords.txt")
#stops = [l.strip() for l in stopwords.readlines()]
stops = set(stopwords.words('english'))
"""lines = [l.strip().strip(string.punctuation).lower().split() for l in testdata.readlines()]"""
lines = headline_titles
"""testdata.close()"""
#stopwords.close()

wordgraph = nx.DiGraph() #initialize directed graph for use, as well as start and end nodes
wordgraph.add_node(-1, attribute="start")
wordgraph.add_node(9999, attribute="end")
"""for i in range(len(lines[0])):
    wordgraph.add_node(i, name=lines[0][i], attribute=wn.synsets(lines[0][i])[0].pos())
for i in range(len(lines[0]) - 1):
    wordgraph.add_edge(i, i + 1)
wordgraph.add_edge(-1, 0)
wordgraph.add_edge(len(lines[0])-1, 9999)"""

flatten = lambda l: [item for sublist in l for item in sublist] #pre-define flatten() function for later use
def addweight(u, v, nweight=0):
    """
    Checks whether an edge u-v exists, and if it does increment its weight by nweight
    If not, simply add the edge u-v
    """
    if wordgraph.has_edge(u, v):
        wordgraph.add_edge(u, v, weight=wordgraph[u][v]['weight']+nweight)
    else:
        wordgraph.add_edge(u, v, weight=nweight)
for i in range(len(lines)):
    """
    """
    curline = lines[i]
    lastnode = None
    for j in range(len(curline)): #iterate through each line
        #for each word:
        # check if it already exists
        # if so, check whether it exists multiple times
        #  multiple times: use predecessor/successor to compare overlap to find best fit
        #  once: use the existing node
        #       (for stopwords only) Check overlap and add only if there is overlap
        cur = curline[j]
        newid = len(wordgraph.nodes)
        if len(wn.synsets(cur)) > 0:
            wordstat = wn.synsets(cur)[0].pos() #get word categories from NLTK
        else:
            wordstat = None
        print(wordstat)
        #existing = [x for x in wordgraph.nodes if wordgraph[x].get('name') == cur and wordgraph[x].get('attribute') == wordstat]
        existing = []
        for x in wordgraph.nodes:
            #print("wordgraph contents of", x, ":", wordgraph[x])
            if wordgraph.nodes[x].get('name') == cur and wordgraph.nodes[x].get('attribute') == wordstat:
                existing.append(x)
        print(existing)
        overlaps = [None, 0]
        if cur not in stops:
            if len(existing) > 1:
                for node in existing:
                    score = 0
                    preds = flatten([wordgraph.nodes[i]['name'] for i in wordgraph.predecessors(node)])
                    print(preds)
                    succs = flatten([wordgraph.nodes[i]['name'] for i in wordgraph.successors(node)])
                    print(succs)
                    if j - 1 >= 0 and curline[j - 1] in preds:
                        score += 1
                    if j + 1 < len(curline) and curline[j + 1] in succs:
                        score += 1
                    if score >= overlaps[1]:
                        overlaps = [node, score]
                newid = overlaps[0]
            elif len(existing) == 1:
                newid = existing[0]
            elif len(existing) == 0:
                print("adding non stopword")
                wordgraph.add_node(newid, name=cur, attribute=wordstat)
                print("contents of newly added", newid, ":", wordgraph.nodes[newid])
        else:
            #Check for existence, then overlap
            #if there is overlap, then map
            #if not then make new node
            if len(existing) > 1:
                for node in existing:
                    score = 0
                    preds = [wordgraph.nodes[i].get('name') for i in wordgraph.predecessors(node)]
                    print("preds",preds)
                    succs = [wordgraph.nodes[i].get('name') for i in wordgraph.successors(node)]
                    print("succs",succs)
                    if j - 1 >= 0 and curline[j - 1] in preds:
                        score += 1
                    if j + 1 < len(curline) and curline[j + 1] in succs:
                        score += 1
                    if score >= overlaps[1]:
                        overlaps = [node, score]
                newid = overlaps[0]
            elif len(existing) == 1:
                #newid = existing[0]
                score = 0
                preds = [wordgraph.nodes[i].get('name') for i in wordgraph.predecessors(existing[0])]
                print("preds",preds)
                succs = [wordgraph.nodes[i].get('name') for i in wordgraph.successors(existing[0])]
                print("succs",succs)
                if j - 1 >= 0 and curline[j - 1] in preds:
                    score += 1
                if j + 1 < len(curline) and curline[j + 1] in succs:
                    score += 1
                if score > 0:
                    newid = existing[0]
                else:
                    print("adding stopword")
                    wordgraph.add_node(newid, name=cur, attribute=wordstat)
                    print("contents of newly added", newid, ":", wordgraph.nodes[newid])
            elif len(existing) == 0:
                print("adding stopword")
                wordgraph.add_node(newid, name=cur, attribute=wordstat)
                print("contents of newly added", newid, ":", wordgraph.nodes[newid])
                
        if not lastnode:
                addweight(-1, newid, nweight=1)
        else:
            addweight(lastnode, newid, nweight=1)
        lastnode = newid
    addweight(lastnode, 9999, nweight=1)
#Invert weights
for edge in wordgraph.edges:
    wordgraph.edges[edge]['weight'] = 1/wordgraph.edges[edge]['weight']
plt.figure(3,figsize=(12,12)) 
print(nx.get_node_attributes(wordgraph, 'attribute'))
nx.draw(wordgraph, labels = nx.get_node_attributes(wordgraph, 'name'), k=0.3*1/math.sqrt(len(wordgraph.nodes())))

def navigate(curnode = -1, result=[]): #simple limited DFS
    print("curnode", curnode)
    print("curword", wordgraph.nodes[curnode].get('name'))
    if curnode != -1 and curnode != 9999:
        result.append(curnode)
    if len(list(filter(lambda x: wordgraph.nodes[x].get('attribute') == 'v', result))) >= 1 and curnode == 9999 and len(result) >= 6:
        return result
    for i in sorted(wordgraph.successors(curnode), key=lambda succ:wordgraph[curnode][succ].get('weight')):
        if i not in result:
            print("weight", wordgraph[curnode][i].get('weight'))
            temp = navigate(i, result)
            if temp != None:
                return temp
        
        
#for edge in wordgraph.edges:
#    print(wordgraph.nodes[edge[0]].get("name"), "to", wordgraph.nodes[edge[1]].get("name"), "is weight", wordgraph[edge[0]][edge[1]].get('weight'))
result = navigate()
print(result)
resultwords = [wordgraph.nodes[i].get('name') for i in result]
print(resultwords)


import spacy
import networkx as nx
import matplotlib.pyplot as plt
import math
from nltk.corpus import wordnet as wn
import random
"""
Game plan: 
    1. Generate seed sentence according to grammar rules
    2. Parse the seed sentence with Spacy, then convert to a NetworkX tree (possible alternative: NLTK tree?)
    3. Drop or add elements at random (as long as they fit the grammar rules)
    4. Make the generated sentence the new seed sentence, then repeat from step 2
    
For now, don't implement category-based word selection- if generation works at all then that's great

Current problems:
    -WordNet isn't expansive enough for current purposes, alternatives?
    -
"""
"""
Note regarding NX trees:
    Predecessors = children
    Successors = parent
"""
def nx_tree(spacy_tree):
    """
    Traverse a Spacy tree and copy its contents to a NetworkX tree
    """
    root = [token for token in spacy_tree if token.head == token][0]
    tree = nx.DiGraph()
    
    def traverse(node, parent=None, childpos=None):
        newnode_id = tree.number_of_nodes()
        tree.add_node(newnode_id, content=node.text)
        if parent is not None:
            if childpos is not None:
                tree.add_edge(newnode_id, parent, dependency=node.dep_, pos=childpos)
            else:
                tree.add_edge(newnode_id, parent, dependency=node.dep_)
        if node.n_lefts > 0:
            for child in node.lefts:
                traverse(child, newnode_id, 'left')
        if node.n_rights > 0:
            for child in node.rights:
                traverse(child, newnode_id, 'right')
    traverse(root)
    return tree


flatten = lambda l: [item for sublist in l for item in sublist] #flatten is such a useful tool to have around

allverbs = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('v')])
allnouns = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('n')])
alladj = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('a')])
prepfile = open("prepositions.txt", "r")
allpreps = [line for line in prepfile.read().splitlines()]
prepfile.close()

def gensentence(root_sentence=None): #Define a function for this so we can recursively add on composite sentences
    seed_tree = nx.DiGraph()
    seed_tree.add_node(0, content=random.choice(allverbs)) #give root verb; later on add random chance for root noun instead?
    nsubj = random.choice(allnouns)
    amod1 = random.choice(alladj)
    amod2 = random.choice(alladj)
    pobj = random.choice(allnouns)
    prep = random.choice(allpreps)
    det = "the"
    nmod = random.choice(range(1, 11))
    #WIP

def buildsentence(tree):
    """
    Build sentence from (NetworkX) parse tree
    """
    
    root = 0#[node for node in tree.nodes() if len(list(tree.successors(node))) == 0][0]
    sentencelist = [tree.nodes[root].get("content")]
    def populate(idx, nodeid):
        print("node being populated", nodeid)
        print("predecessors", list(tree.predecessors(nodeid)))
        for child in tree.predecessors(nodeid):
            if tree.nodes[child].get('pos') is not None:
                print("cur child", child)
                if tree.nodes[child].get('pos') == 'left':
                    sentencelist.insert(idx - 1, tree.nodes[child].get('content'))
                    populate(idx - 1, child)
                else:
                    sentencelist.insert(idx + 1, tree.nodes[child].get('content'))
                    populate(idx + 1, child)
    populate(0, 0)
    print(sentencelist)
    sentencelist[0] = sentencelist[0].capitalize()
    return " ".join(sentencelist)
        
    


    
nlp = spacy.load("en_core_web_sm") #for testing nx_tree
temp = nlp("The quick brown fox jumps over the lazy dog")
nxify = nx_tree(temp)

print(buildsentence(nxify))

"""
plt.figure(3,figsize=(12,12)) 
print(nx.get_node_attributes(nxify, 'content'))
print(nx.get_edge_attributes(nxify, 'dependency'))
nx.draw(nxify, labels = nx.get_node_attributes(nxify, 'content'), k=0.3*1/math.sqrt(len(nxify.nodes())))
"""
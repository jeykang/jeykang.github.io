#Implement rules for sentence generation
#First, generate seed words for each position in the following:
#Proper noun - Verb - Article - (Up to 2 adjectives) - Proper noun OR noun - (Optional: preposition - Proper noun OR article - noun)
#Decide whether proper noun or noun will be used
#Decide each word specifically to form a seed sentence
#Use WordNet hierarchy to form variations of the seed sentence
#For outliers, TO BE IMPLEMENTED LATER

from nltk.corpus import wordnet as wn
import random
from pattern.en import lexeme
#Useful note for lexeme: [normal, present, present (ongoing), past, ...]
def pasttense(word):
    temp = lexeme(word)
    if len(temp) >= 4:
        return temp[3]
    else:
        return temp[0]
flatten = lambda l: [item for sublist in l for item in sublist] #pre-define flatten() function for later use
#BIG flaw! Wordnet has no processing capability for prepositions, do they need to be manually defined?
#Problem with current method is that there is no category tagging for prepositions- among and from are clearly different,
#but there is no way to tag them as such without manual action
prepfile = open("prepositions.txt", "r")
prepositions = [line for line in prepfile.read().splitlines()]
prepfile.close()
articles = ["a", "the", "an"]
#Category to pull nouns from- way to randomly generate or define per run?
category1 = "object"
category2 = "person"
#Generate noun list from given category

cat1nouns = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('n') if category1 in syn.lexname()])
cat2nouns = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('n') if category2 in syn.lexname()])
    
#Verb category & definition
verbdef = "give"
verbcat = "possess"

allverbs = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('v') if verbcat in syn.lexname() and verbdef in syn.definition()])

#Adj 
catadj = "all"
alladj = flatten([[lemma.name() for lemma in syn.lemmas()] for syn in wn.all_synsets('a') if catadj in syn.lexname()])
#Is it possible to generate proper nouns as well? For now just give it
propnoun_per = "James Cameron"
propnoun_obj = "Oscar Award"
propnoun_rel = "Committee"

usenoun_obj = random.choice([0, 1]) #choose whether to use a noun for obj or not
usenoun_rel = random.choice([0, 1]) #choose whether to use a noun for the related 

noun_obj = random.choice(cat1nouns)
noun_rel = random.choice(cat2nouns)

verb = random.choice(allverbs)


#Num. of sentences needed for generation
numsentences = 10000
#generated sentences- seed with initial sentence - (IMPLEMENT MULTIPLE ADJ LATER) (ALSO IMPLEMENT COMPOUND SENTS (w/ AND))
if usenoun_obj:
    if usenoun_rel:
        articles = ["a", "a"]
        gensentences = [(propnoun_per, verb, articles[0], random.choice(alladj), noun_obj, random.choice(prepositions), articles[1], noun_rel)]
    else:
        articles = ["a", "the"]
        gensentences = [(propnoun_per, verb, articles[0], random.choice(alladj), noun_obj, random.choice(prepositions), articles[1], propnoun_rel)]


else:
    if usenoun_rel:
        articles = ["the", "a"]
        gensentences = [(propnoun_per, verb, articles[0], random.choice(alladj), propnoun_obj, random.choice(prepositions), articles[1], noun_rel)]
    else:
        articles = ["the", "the"]
        gensentences = [(propnoun_per, verb, articles[0], random.choice(alladj), propnoun_obj, random.choice(prepositions), articles[1], propnoun_rel)]
for _ in range(numsentences):
    use_long = random.choice([0, 1]) #random probability to generate long or short sentence
    lastsent = gensentences[-1] #use previous sentence to generate variation
    if len(wn.synsets(lastsent[1])) > 0:
        newverb = random.choice(wn.synsets(lastsent[1])).lemmas()[0].name()
    else:
        newverb = gensentences[0][1]
    #newadj = random.choice(wn.synsets(lastsent[3])[0].hypernyms()[0].hyponyms()).lemmas()[0].name()
    newadj = random.choice(wn.synsets(lastsent[3])).lemmas()[0].name()
    if usenoun_obj:
        if len(wn.synsets(lastsent[4])) > 0:
            try:
                newobj = random.choice(wn.synsets(lastsent[4])[0].hypernyms()[0].hyponyms()).lemmas()[0].name()
            except IndexError:
                newobj = random.choice(wn.synsets(lastsent[4])).lemmas()[0].name()
        else:
            newobj = gensentences[0][4]
    else:
        newobj = lastsent[4]
    if use_long:
        newprep = random.choice(prepositions)
        if len(lastsent) > 5: 
            if usenoun_rel:
                if len(wn.synsets(lastsent[7])) > 0:
                    newrel = random.choice(wn.synsets(lastsent[7])[0].hypernyms()[0].hyponyms()).lemmas()[0].name()
                else:
                    newrel = gensentences[0][7]
            else:
                newrel = lastsent[7]
        else:
            if usenoun_rel: #If last sentence was short then use first sentence as reference
                #This creates gradual variance
                if len(wn.synsets(gensentences[0][7])) > 0:
                    newrel = random.choice(wn.synsets(gensentences[0][7])[0].hypernyms()[0].hyponyms()).lemmas()[0].name()
                else:
                    newrel = gensentences[0][7]
            else:
                newrel = gensentences[0][7]
        gensentences.append((propnoun_per, newverb, articles[0], newadj, newobj, newprep, articles[1], newrel))
    else:
        gensentences.append((propnoun_per, newverb, articles[0], newadj, newobj))

gensentences = set(gensentences)
completesentences = []
for sent in gensentences:
    if len(sent) > 5:
        sentstring = sent[0] + " " + pasttense(sent[1]) + " " + sent[2] + " " + sent[3] + " " + sent[4] + " " + sent[5] + " " + sent[6] + " " + sent[7]
    else:
        sentstring = sent[0] + " " + pasttense(sent[1]) + " " + sent[2] + " " + sent[3] + " " + sent[4]
    completesentences.append(sentstring)
print(completesentences)
       
        
                
        
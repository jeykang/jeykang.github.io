import csv

def preprocess(path):
    with open(path, 'r') as yearpop:
        popdict = {}
        csvpop = csv.reader(yearpop, delimiter=',')
        count = 0
        for row in csvpop:
            if count == 0:
                topline = row
            else:
                popline = row
                popdict[popline[1]] = []
                for i in range(len(popline[2:])):
                    if topline[i+2] != '' and popline[i+2] != '':
                        popdict[popline[1]].append((float(topline[i + 2]), float(popline[i + 2])))
                    elif topline[i+2] == '':
                        popdict[popline[1]].append((0, float(popline[i + 2])))
                    elif popline[i+2] == '':
                        popdict[popline[1]].append((float(topline[i + 2]), 0))
            count += 1
        return popdict

def codetoname(path):
    with open(path, 'r') as yearpop:
        namedict = {}
        csvpop = csv.reader(yearpop, delimiter=',')
        count = 0
        for row in csvpop:
            if count > 0:
                namedict[row[1]] = row[0] 
            count += 1
        return namedict
    
def years(path):
    with open(path, 'r') as yearpop:
        csvpop = csv.reader(yearpop, delimiter=',')
        count = 0
        for row in csvpop:
            if count == 0:
                return [int(i) for i in row[2:]]
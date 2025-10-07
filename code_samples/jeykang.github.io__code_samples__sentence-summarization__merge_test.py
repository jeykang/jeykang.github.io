"""
Implementation of the algorithm detailed in 
https://pdfs.semanticscholar.org/f710/ed4b2bd4071cb33445ccb76eb611a4851784.pdf
"""

Merge(t1, t2) :-
    if isLeafCondition (t1,t2) then
        return findAppropriateLeaf(t1,t2);
    else
        host ¬ chooseHost(t1,t2); #host is always c^n
        pairings ¬ pairDaughters(t1, t2));
        if isEmpty(pairings) then
            return null;
        else
            for 〈p1, p2〉 ∈ pairings
                newDaughter ¬ merge(p1, p2);
                if newDaughter is not null then
                    addDaughter(host, newDaughter);
                else
                    if not sufficientlyMerged (〈p1, p2〉) then
                        return null;
            return host;
        
pairDaughters (t1, t2) :-
    for R ∈ ruleLibrary
        if evaluateConstraints(R, t1, t2) then
            for 〈d1, d2〉 ∈ potentialPairings(R, t1, t2)
                if isCompatible(d1, d2) then
                    push(pairedDaughters, 〈d1, d2〉);
            for d ∈ (daughters(t1) ∪ daughters(t1))
                if d not present in pairedDaughters then
                    push(pairedDaughters, 〈d, d〉);
            return pairedDaughters;
    return empty list;

def merge(cn, sn):
    if : #both are leaves? one is a leaf? not sure
        return 
    else:
        host = cn
        pairings = pairDaughters(cn, sn)
        if len(pairings) == 0:
            return
        else:
            for pair in pairings:
                newDaughter = merge(pair[0], pair[1])
                if newDaughter is not None:
                    addDaughter(host, newDaughter)
                else:
                    if not sufficientlyMerged(pair):
                        return
            return host
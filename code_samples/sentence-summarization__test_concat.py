"""
For testing recursive-sentenceconcat (and revisions)
"""

from recursive_sentenceconcat_revision import do_concat
from numpy.random import permutation

f = open("1984.txt")
totlines = [line for line in f.readlines() if line != "\n"]
f.close()

#print(totlines)
#permute totlines arbitrary number of times, and store the results
#total_gen = []
total_gen = totlines
already_done = []
n = 500
"""for j in range(n):
    perm = permutation(totlines).tolist()
    retry = 0 #num of retries
    while perm in already_done and retry < n: #end if there are no more possible permutations
        retry += 1
        perm = permutation(totlines).tolist()
    if retry >= n:
        break
    already_done.append(perm)
    i = 1
    cur_cn = perm[0]
    while i < len(perm):
        cur_cn = do_concat([cur_cn, perm[i]])
        i += 1
    print(cur_cn)
    total_gen.append(cur_cn)"""
    
#Then just run through the generated summaries with do_concat
final = do_concat(total_gen, draw_plot=True, show_logs=True)
#final = do_concat(totlines, draw_plot=True, show_logs=True)
print(final)
#print(total_gen)


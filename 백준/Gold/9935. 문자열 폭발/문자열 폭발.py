import sys

allString = sys.stdin.readline().strip()
pattern = sys.stdin.readline().strip()
pattern_size = len(pattern)
q, pattern_end = [], pattern[pattern_size - 1]

for i in allString :
    q.append(i)
    if i == pattern_end and "".join(q[-pattern_size:]) == pattern :
        del q[-pattern_size:]

if q :
    print("".join(q))
else :
    print("FRULA")

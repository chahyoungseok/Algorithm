from collections import defaultdict

N = int(input())
words, v = [], defaultdict(int)
for _ in range(N) :
    words.append(input())

for word in words :
    s = len(word) - 1
    for a in word :
        v[a] += 10 ** s
        s -= 1

v = sorted(v, key=lambda x : v[x], reverse=True)

sel, total = 9, 0
for a in v :
    for i in range(N):
        words[i] = words[i].replace(a, str(sel))
    sel -= 1

for word in words :
    total += int(word)
print(total)
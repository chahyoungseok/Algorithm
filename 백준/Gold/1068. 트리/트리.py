import sys
from collections import deque

N = int(sys.stdin.readline().strip())
parents = list(map(int, (sys.stdin.readline()).split()))
delete_node = int(sys.stdin.readline().strip())

tree, root = {}, 0
for i in range(N) :
    if parents[i] == -1 :
        root = i
    if parents[i] != delete_node and i != delete_node :
        if parents[i] in tree.keys():
            tree[parents[i]].append(i)
        else:
            tree[parents[i]] = [i]


q, count = deque(), 0

if root != delete_node :
    q.append(root)

while q :
    cur = q.popleft()

    if cur in tree.keys() and tree[cur]:
        for i in tree[cur] :
            q.append(i)
    else :
        count += 1
print(count)

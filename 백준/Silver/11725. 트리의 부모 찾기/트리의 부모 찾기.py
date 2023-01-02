import sys
sys.setrecursionlimit(int(1e9))


class Tree :
    def __init__(self, data, parent, current, answer):
        answer[current] = parent

        for j in data[current] :
            if j != parent :
                Tree(data, current, j, answer)


N = int(sys.stdin.readline().strip())
edges = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    a, b = map(int, (sys.stdin.readline()).split())
    edges[a].append(b)
    edges[b].append(a)

answer = [0 for _ in range(N + 1)]
tree = Tree(edges, None, 1, answer)
for i in range(2, N + 1) :
    print(answer[i])
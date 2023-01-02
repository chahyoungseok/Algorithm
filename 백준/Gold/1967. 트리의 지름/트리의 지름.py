import sys
sys.setrecursionlimit(int(1e8))


class Tree :
    def __init__(self, datas, current):
        self.root = current
        self.child = []
        self.child_dist = []
        if current in datas.keys() :
            childrens = datas[current]
            for i in range(len(childrens)) :
                self.child.append(Tree(datas, childrens[i][0]))
                self.child_dist.append(childrens[i][1])

    def cal_dist(self, dist):
        max_dist = 0
        dist_lists = []
        for i in range(len(self.child)) :
            x, y = self.child[i].cal_dist(self.child_dist[i])
            dist_lists.append(x)
            max_dist = max(max_dist, y)

        dist_lists = sorted(dist_lists, reverse=True)
        one, two = 0, 0
        if len(dist_lists) == 1 :
            one = dist_lists[0]
        elif len(dist_lists) >= 2 :
            one, two = dist_lists[0], dist_lists[1]

        return dist + max(one, two), max(one + two, max_dist)


n = int(sys.stdin.readline().strip())
edges = {}
for _ in range(n - 1) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    if a not in edges.keys() :
        edges[a] = []
    edges[a].append([b, c])

tree = Tree(edges, 1)
g, result = tree.cal_dist(0)
print(result)
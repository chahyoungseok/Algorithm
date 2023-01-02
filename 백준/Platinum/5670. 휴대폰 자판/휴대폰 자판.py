import sys


class Node :
    def __init__(self, key, data=None):
        self.key = key
        self.data = data
        self.children = {}
        self.dist = 1


class Trie :
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head
        for char in string :
            if char not in current_node.children :
                current_node.children[char] = Node(char)
            current_node = current_node.children[char]
        current_node.data = string

    def search(self):
        current_node = list(self.head.children.values())
        next_node = []

        while True :
            for node in current_node :
                if node.data :
                    global dist_list
                    dist_list.append(node.dist)

                for c in node.children.values() :
                    c.dist = node.dist

                if len(node.children) > 1 or node.data:
                    for c in node.children.values() :
                        c.dist += 1
                next_node.extend(list(node.children.values()))

            if len(next_node) == 0 :
                break
            else :
                current_node = next_node
                next_node = []


while True :
    try :
        N = int(sys.stdin.readline().strip())
    except :
        break

    trie = Trie()
    for _ in range(N) :
        trie.insert(sys.stdin.readline().strip())

    dist_list = []
    trie.search()
    print("%.2f" % (sum(dist_list) / N))

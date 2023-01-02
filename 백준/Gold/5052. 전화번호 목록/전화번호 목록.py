import sys


class Node(object) :
    def __init__(self, key, data=None):
        self.key = key
        self.data = data
        self.childrens = {}


class Trie :
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head

        for char in string :
            if char not in current_node.childrens :
                current_node.childrens[char] = Node(char)
            current_node = current_node.childrens[char]
            if current_node.data :
                return False
        current_node.data = string
        return True


t = int(sys.stdin.readline().strip())

for _ in range(t) :
    n = int(sys.stdin.readline().strip())
    trie, numbers, state = Trie(), [], True

    for _ in range(n) :
        numbers.append(sys.stdin.readline().strip())

    numbers = sorted(numbers, key=lambda x : (len(x), x))

    for number in numbers :
        if not trie.insert(number) :
            state = False
            break

    if state :
        print("YES")
    else :
        print("NO")

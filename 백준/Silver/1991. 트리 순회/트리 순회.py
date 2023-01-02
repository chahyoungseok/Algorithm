import sys


class Tree :
    def __init__(self, datas, current):
        self.root = current
        if datas[current][0] == '.' :
            self.left_node = None
        else :
            self.left_node = Tree(datas, datas[current][0])

        if datas[current][1] == '.':
            self.right_node = None
        else:
            self.right_node = Tree(datas, datas[current][1])


N = int(sys.stdin.readline().strip())

edges_dict = {}
for _ in range(N):
    c, l, r = map(str, (sys.stdin.readline()).split())
    edges_dict[c] = [l, r]

tree = Tree(edges_dict, 'A')
pre_list, in_list, post_list = [], [], []


def preorder(tree, pre_list) :
    pre_list.append(tree.root)
    if tree.left_node :
        preorder(tree.left_node, pre_list)
    if tree.right_node :
        preorder(tree.right_node, pre_list)


def inorder(tree, in_list) :
    if tree.left_node :
        inorder(tree.left_node, in_list)
    in_list.append(tree.root)
    if tree.right_node :
        inorder(tree.right_node, in_list)


def postorder(tree, post_list) :
    if tree.left_node :
        postorder(tree.left_node, post_list)
    if tree.right_node :
        postorder(tree.right_node, post_list)
    post_list.append(tree.root)


preorder(tree, pre_list)
inorder(tree, in_list)
postorder(tree, post_list)

for i in range(N) :
    print(pre_list[i], end="")
print()
for i in range(N) :
    print(in_list[i], end="")
print()
for i in range(N) :
    print(post_list[i], end="")
print()
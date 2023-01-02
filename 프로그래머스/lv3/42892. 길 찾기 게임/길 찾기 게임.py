import sys
sys.setrecursionlimit(10 ** 6)


class Tree:
    def __init__(self, datas):
        self.root = max(datas, key=lambda x: x[1])
        self.index = self.root[2]

        left_ = list(filter(lambda x: x[0] < self.root[0], datas))
        right_ = list(filter(lambda x: x[0] > self.root[0], datas))

        if left_:
            self.left_tree = Tree(left_)
        else:
            self.left_tree = None

        if right_:
            self.right_tree = Tree(right_)
        else:
            self.right_tree = None


def solution(nodeinfo):
    pre_list, post_list = [], []
    for i in range(len(nodeinfo)):
        nodeinfo[i].append(i + 1)

    tree = Tree(nodeinfo)

    def pre_order(tree, pre_list):
        pre_list.append(tree.index)
        if tree.left_tree :
            pre_order(tree.left_tree, pre_list)
        if tree.right_tree :
            pre_order(tree.right_tree, pre_list)

    def post_order(tree, post_list):
        if tree.left_tree:
            post_order(tree.left_tree, post_list)
        if tree.right_tree:
            post_order(tree.right_tree, post_list)
        post_list.append(tree.index)

    pre_order(tree, pre_list)
    post_order(tree, post_list)

    return [pre_list, post_list]
import copy


class Tree:
    def __init__(self, index, info, edges):
        self.root = index
        self.sw = info[index]

        childrens = []
        for parent, children in edges:
            if parent == index:
                childrens.append(children)

        if not childrens:
            self.left_node = None
            self.right_node = None
        elif len(childrens) == 1:
            self.left_node = Tree(childrens[0], info, edges)
            self.right_node = None
        elif len(childrens) == 2:
            self.left_node = Tree(childrens[0], info, edges)
            self.right_node = Tree(childrens[1], info, edges)


def solution(info, edges):
    root_tree = Tree(0, info, edges)
    tree_list = []
    if root_tree.left_node :
        tree_list.append(root_tree.left_node)
    if root_tree.right_node :
        tree_list.append(root_tree.right_node)

    def dfs(tree_list, ship, wolf, max_ship):
        for i in range(len(tree_list)):
            copy_list = copy.deepcopy(tree_list)
            copy_list.pop(i)
            copy_ship, copy_wolf = ship, wolf

            if tree_list[i].sw == 0:
                copy_ship += 1
                if copy_ship > max_ship:
                    max_ship = copy_ship
            else:
                copy_wolf += 1

            if copy_wolf >= copy_ship:
                continue

            if tree_list[i].left_node :
                copy_list.append(tree_list[i].left_node)
            if tree_list[i].right_node :
                copy_list.append(tree_list[i].right_node)

            max_ship = max(max_ship, dfs(copy_list, copy_ship, copy_wolf, max_ship))

        return max_ship

    return dfs(tree_list, 1, 0, 1)
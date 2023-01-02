from collections import defaultdict


class Node:
    def __init__(self, key):
        self.key = key
        self.cnt = defaultdict(int)
        self.children = {}


class Trie:
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head

        for char in string:
            if char not in current_node.children:
                current_node.children[char] = Node(char)
            current_node.cnt[len(string)] += 1
            current_node = current_node.children[char]

    def starts_with(self, prefix, query_len):
        current_node = self.head

        for p in prefix:
            if p in current_node.children:
                current_node = current_node.children[p]
            else:
                return 0

        if query_len in current_node.cnt.keys() :
            return current_node.cnt[query_len]
        else :
            return 0


def solution(words, queries):
    trie_start, trie_end, answer = Trie(), Trie(), []
    dic = {}
    for word in words:
        trie_start.insert(word)

        word = list(word)
        word.reverse()
        trie_end.insert(word)

    for query in queries:
        query_len = len(query)
        if query in dic.keys() :
            answer.append(dic[query])
        else :
            if query[0] == "?":
                query = query.replace("?", "")
                query = list(query)
                query.reverse()

                answer.append(trie_end.starts_with(query, query_len))
            else:
                query = query.replace("?", "")
                answer.append(trie_start.starts_with(query, query_len))

    return answer
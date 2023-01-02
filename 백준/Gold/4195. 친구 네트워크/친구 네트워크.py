#1:36
import sys


def find(friend_x) :
    if parent[friend_x] != friend_x :
        return find(parent[friend_x])
    return friend_x


def union(friend_a, friend_b) :
    friend_a = find(friend_a)
    friend_b = find(friend_b)

    if friend_a != friend_b :
        if friend_a > friend_b:
            parent[friend_b] = friend_a
            count[friend_a] += count[friend_b]
            print(count[friend_a])
        else:
            parent[friend_a] = friend_b
            count[friend_b] += count[friend_a]
            print(count[friend_b])
    else :
        print(count[friend_a])


T = int(sys.stdin.readline().strip())

for _ in range(T) :
    F = int(sys.stdin.readline().strip())

    parent, count = {}, {}
    for _ in range(F) :
        friend_a, friend_b = map(str, sys.stdin.readline().split())
        if friend_a not in parent.keys() :
            parent[friend_a] = friend_a
            count[friend_a] = 1
        if friend_b not in parent.keys() :
            parent[friend_b] = friend_b
            count[friend_b] = 1

        union(friend_a, friend_b)

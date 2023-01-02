from collections import deque


def solution(cacheSize, cities):
    if cacheSize == 0 :
        return len(cities) * 5
    q, total = deque(), 0
    for city in cities:
        name = city.lower()
        if cacheSize > len(q):
            if name in q :
                q.remove(name)
                total += 1
            else :
                total += 5
            q.append(name)
            continue
        if name in q:
            q.remove(name)
            total += 1
        else:
            q.popleft()
            total += 5

        q.append(name)

    return total
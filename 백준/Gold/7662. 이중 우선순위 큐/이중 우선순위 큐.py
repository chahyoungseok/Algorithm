import sys, heapq

T = int(sys.stdin.readline().strip())

for _ in range(T) :
    k = int(sys.stdin.readline().strip())

    exists = [True for _ in range(k + 1)]
    exists[k] = False
    min_q, max_q, in_count = [], [], 0
    for i in range(k) :
        case, value = map(str, (sys.stdin.readline()).split())
        value = int(value)
        if case == "D" :
            if value == 1:
                index = k
                while max_q and not exists[index] :
                    value, index = heapq.heappop(max_q)
                exists[index] = False
            else:
                index = k
                while min_q and not exists[index]:
                    value, index = heapq.heappop(min_q)
                exists[index] = False
        else :
            heapq.heappush(min_q, (value, i))
            heapq.heappush(max_q, (-value, i))
            in_count += 1
    if k - in_count != sum(exists) :
        max_value, min_value = -int(1e20), int(1e20)
        max_index, min_index = k, k
        while max_q and not exists[max_index]:
            max_value, max_index = heapq.heappop(max_q)

        while min_q and not exists[min_index]:
            min_value, min_index = heapq.heappop(min_q)

        print(str(-max_value) + " " + str(min_value))
    else :
        print("EMPTY")
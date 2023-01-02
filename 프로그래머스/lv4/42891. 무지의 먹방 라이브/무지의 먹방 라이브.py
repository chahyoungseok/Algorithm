import heapq


def solution(food_times, k):
    current_time, food_count, food_heap = 0, len(food_times), []
    previous, sum_time = 0, 0

    for i in range(food_count) :
        heapq.heappush(food_heap, (food_times[i], i + 1))

    while True:
        time, num = heapq.heappop(food_heap)
        current_time = ((time - previous) * food_count)
        sum_time += current_time

        if k >= sum_time :
            food_count -= 1
            previous = time
            if food_count <= 0:
                return -1
        else :
            food_heap.append((time, num))
            food_heap = sorted(food_heap, key=lambda x:x[1])
            return food_heap[(k - sum_time + current_time) % food_count][1]
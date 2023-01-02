import heapq

N = int(input())
card_bundle = []

for _ in range(N) :
    heapq.heappush(card_bundle, int(input()))

sum_compare = heapq.heappop(card_bundle) * (N - 1)

for i in range(1,N) :
    sum_compare += heapq.heappop(card_bundle) * (N - i)

print(sum_compare)
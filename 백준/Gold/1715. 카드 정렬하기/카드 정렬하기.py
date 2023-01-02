import sys, heapq

N = int(input())
cards = []
for _ in range(N) :
    heapq.heappush(cards, int(sys.stdin.readline()))

total = 0
while len(cards) != 1 :
    a = heapq.heappop(cards)
    b = heapq.heappop(cards)
    c = a + b
    total += c
    heapq.heappush(cards, c)
print(total)
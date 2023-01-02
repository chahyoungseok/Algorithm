N = int(input())
P = list(map(int, input().split()))

time, total_time = 0, 0
P = sorted(P)

for i in P :
    time += i
    total_time += time

print(total_time)
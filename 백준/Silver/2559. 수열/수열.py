N, K = map(int, input().split())
temperate = list(map(int, input().split()))
t_a = [sum(temperate[:K])]

for i in range(N - K) :
    t_a.append(t_a[i] - temperate[i] + temperate[K + i])

print(max(t_a))
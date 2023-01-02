N = int(input())

fear_rate = list(map(int, input().split()))
rate_array = [0 for _ in range(0, N)]
group_count = 0

for i in range(0, N) :
    rate_array[fear_rate[i]] += 1

for i in range(1, N) :
    group_count += rate_array[i] // i

print(group_count)
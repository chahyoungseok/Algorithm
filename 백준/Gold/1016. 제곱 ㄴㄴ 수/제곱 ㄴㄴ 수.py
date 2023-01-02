import math

n, m = map(int, input().split())
states = [1] * (m - n + 1)
squares = [s ** 2 for s in range(2, int(math.sqrt(m)) + 1)]

for i in squares :
    re = n % i
    if re == 0 :
        re = i
    idx = i - re
    while m - n + 1 > idx:
        states[idx] = 0
        idx += i

print(sum(states))

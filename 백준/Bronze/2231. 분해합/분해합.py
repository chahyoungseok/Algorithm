def decompose(n) :
    total = n
    n = str(n)

    for i in n :
        total += int(i)

    return total


N = int(input())
initial_num = 0

for i in range(N) :
    if decompose(i) == N :
        initial_num = i
        break

print(initial_num)
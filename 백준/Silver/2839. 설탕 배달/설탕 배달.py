N = int(input())
total = 0
for i in range(int(N / 5), -1, -1) :
    result = N - (i * 5)
    if result % 3 == 0 :
        total = i + result // 3
        break
    elif result == N :
        total = -1
        break

print(total)
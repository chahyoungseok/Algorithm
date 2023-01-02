import sys

N = int(input())
numbers = []
for _ in range(N) :
    numbers.append(int(sys.stdin.readline()))

numbers = sorted(numbers)

total, i, j = 0, 0, N - 1
while True :
    if N > i + 1 and 2 > numbers[i + 1]:
        if numbers[i + 1] < 0 :
            total += numbers[i] * numbers[i + 1]
            i += 2
        elif numbers[i + 1] == 0 :
            total += 0
            i += 2
        elif numbers[i + 1] == 1 :
            total += numbers[i] + numbers[i + 1]
            i += 2
    else :
        break

while True:
    if j - 1 >= i and numbers[j - 1] > 1:
        total += numbers[j] * numbers[j - 1]
        j -= 2
    else :
        break

if i < j :
    total += numbers[j] + numbers[i]
elif i == j :
    total += numbers[i]
print(total)
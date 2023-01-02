import sys

N, L = map(int, input().split())
w_list = list(map(int, (sys.stdin.readline()).split()))
w_list = sorted(w_list)
index, total = 0, 0

while N > index :
    e = w_list[index] + L - 1
    while N > index and e >= w_list[index] :
        index += 1

    total += 1

print(total)
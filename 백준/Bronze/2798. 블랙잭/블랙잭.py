from itertools import combinations

N, M = map(int, input().split())
cards = list(map(int, input().split()))
close_M = int(1e9)

for combin in combinations(cards, 3) :
    combin_sum = sum(combin)
    if M >= combin_sum and close_M > M - combin_sum:
        close_M = M - combin_sum

print(M - close_M)
N, M = map(int, input().split())

money_case, total = [], 0
for _ in range(N) :
    money_case.append(int(input()))

money_case.reverse()

for money in money_case :
    sel = M // money
    total += sel
    M -= sel * money

print(total)
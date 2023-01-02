N = int(input())
money_unit = list(map(int, input().split()))
money_unit.sort()

target = 1

for x in money_unit :
    if target < x :
        break
    target += x

print(target)
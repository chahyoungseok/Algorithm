N, M = map(int, input().split())
riceCake = list(map(int, input().split()))

result = 0
start = 0
end = max(riceCake)

while start <= end :
    remainCake = 0
    mid = (start + end) // 2

    for i in riceCake :
        if i > mid :
            remainCake += i - mid

    if remainCake > M :
        result = mid
        start = mid + 1
    elif remainCake == M :
        result = mid
        break
    else :
        end = mid - 1

print(result)
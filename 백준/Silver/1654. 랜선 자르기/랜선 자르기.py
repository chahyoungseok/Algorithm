lan_list = []
K, N = map(int, input().split())
for _ in range(K) :
    lan_list.append(int(input()))


def binary_search(target, start, end) :
    mid, sum = (start + end) // 2, 0
    if start > end :
        return end

    for lan in lan_list :
        sum += lan // mid

    if target > sum :
        return binary_search(target, start, mid - 1)
    elif target <= sum :
        return binary_search(target, mid + 1, end)

print(binary_search(N, 1, max(lan_list)))
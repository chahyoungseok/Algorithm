def binary_search(arr, target, start, end, standard) :
    if start > end :
        return -1
    mid = (start + end) // 2

    if standard - arr[mid] < target :
        return binary_search(arr, target, start, mid - 1, standard)
    elif standard - arr[mid] > target :
        return binary_search(arr, target, mid + 1, end, standard)
    else :
        return mid


N = int(input())

if N == 1 :
    print(0)
else :
    total = 0
    state = [True for _ in range(N + 1)]
    for i in range(2, int((N + 1) ** 0.5) + 1):
        if state[i]:
            for j in range(i + i, N + 1, i):
                state[j] = False
    primeList = [i for i in range(2, N + 1) if state[i]]

    continuous = [0]
    for i in range(len(primeList)):
        continuous.append(primeList[i] + continuous[i])

    continuous_len = len(continuous)
    for i in range(1, len(continuous)):
        if N > continuous[i] :
            continue

        if binary_search(continuous, N, 0, continuous_len - 1, continuous[i]) != -1 :
            total += 1

    print(total)
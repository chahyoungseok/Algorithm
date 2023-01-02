import copy


def makePrime(number) :
    states = [True] * number
    for i in range(2, int(number ** 0.5)) :
        if states[i] :
            for j in range(i + i, number, i) :
                states[j] = False
    return [i for i in range(2, number) if states[i]]


def dfs(number) :
    if visited[number] :
        return False
    visited[number] = True

    for i in odd :
        if copy_list[i] + copy_list[even[number - 1]] in primeNumbers :
            if matched[i] == 0 or dfs(matched[i]) :
                matched[i] = number
                return True
    return False


N = int(input())
numbers = list(map(int, input().split()))
first_number, primeNumbers = numbers[0], makePrime(2000)
result_arr, even_odd = [], True

for i in range(1, N) :
    if first_number + numbers[i] in primeNumbers :
        result, state = [], True
        copy_list = copy.deepcopy(numbers)
        copy_list.pop(i)
        copy_list.pop(0)

        matched = [0 for _ in range(len(copy_list) + 1)]
        odd, even = [], []
        for j in range(len(copy_list)) :
            if copy_list[j] % 2 == 0 :
                even.append(j)
            else :
                odd.append(j)

        if len(even) != len(odd) :
            even_odd = False
            break

        for j in range(1, N // 2) :
            visited = [False] * (len(copy_list) + 1)
            dfs(j)

        total = 0
        for j in matched :
            if j != 0 :
                total += 1
        if total == (N // 2) - 1 :
            result_arr.append(numbers[i])


if not result_arr or not even_odd:
    print(-1)
else :
    result_arr = sorted(result_arr)
    for i in result_arr :
        print(i, end=" ")
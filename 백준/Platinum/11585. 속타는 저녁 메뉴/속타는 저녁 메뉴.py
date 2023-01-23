import sys


def kmp(pattern, allString) :
    pattern_size = len(pattern)
    table = [0 for _ in range(pattern_size)]
    i = 0

    for j in range(1, pattern_size) :
        while i > 0 and pattern[i] != pattern[j] :
            i = table[i - 1]
        if pattern[i] == pattern[j] :
            i += 1
            table[j] = i

    result = []
    i = 0

    for j in range(len(allString)) :
        while i > 0 and pattern[i] != allString[j] :
            i = table[i - 1]
        if pattern[i] == allString[j] :
            i += 1
            if i == pattern_size :
                result.append(j - i + 1)
                i = table[i - 1]

    return len(result)


N = int(sys.stdin.readline().strip())

goal = list(map(str, (sys.stdin.readline()).split()))
roulette = list(map(str, (sys.stdin.readline()).split()))

roulette += roulette[:-1]
c = kmp(goal, roulette)

print("1/" + str(int(N/c)))
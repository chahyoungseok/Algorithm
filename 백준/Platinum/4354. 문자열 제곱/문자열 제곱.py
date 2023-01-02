import sys


def kmp(pattern) :
    pattern_size = len(pattern)
    table = [0 for _ in range(pattern_size)]
    i = 0

    for j in range(1, pattern_size) :
        while i > 0 and pattern[i] != pattern[j] :
            i = table[i - 1]
        if pattern[i] == pattern[j] :
            i += 1
            table[j] = i

    return table


while True :
    data = sys.stdin.readline().strip()
    if data == "." :
        break

    result = kmp(data)
    if len(data) % (len(data) - result[-1]) == 0 :
        print(len(data) // (len(data) - result[-1]))
    else :
        print(1)
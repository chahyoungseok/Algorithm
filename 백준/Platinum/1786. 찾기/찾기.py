import sys


def kmp(all_string, pattern):
    pattern_size = len(pattern)
    table = [0 for _ in range(pattern_size)]

    i = 0
    for j in range(1, pattern_size):
        while i > 0 and pattern[i] != pattern[j]:
            i = table[i - 1]
        if pattern[i] == pattern[j]:
            i += 1
            table[j] = i

    result = []
    i = 0
    for j in range(len(all_string)):
        while i > 0 and pattern[i] != all_string[j]:
            i = table[i - 1]
        if pattern[i] == all_string[j]:
            i += 1
            if pattern_size == i:
                result.append(j - i + 1)
                i = table[i - 1]

    return result


T = input()
P = input()

result = kmp(T, P)

print(len(result))
for lo in result :
    print(lo + 1, end=" ")
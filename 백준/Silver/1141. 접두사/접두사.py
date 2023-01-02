import sys


def check(a, b) :
    for w in range(len(a), len(b) + 1) :
        if b[:w] == a :
            return True
    return False


N = int(sys.stdin.readline().strip())
words = []
for _ in range(N) :
    words.append(sys.stdin.readline().strip())

words = sorted(words, key=lambda x : len(x))
total = 0
for i in range(N) :
    state = True
    for j in range(i + 1, N) :
        if check(words[i], words[j]) :
            state = False
            break
    if state :
        total += 1
print(total)
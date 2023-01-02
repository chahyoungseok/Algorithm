import sys


def makePrimeList(number) :
    state = [True] * number

    for i in range(2, int(number ** 0.5)) :
        if state[i] :
           for j in range(i + i, number, i) :
               state[j] = False

    return [i for i in range(2, number) if state[i]]


N = int(input())
prime_list = list(map(int, (sys.stdin.readline()).split()))
total, standard = 0, makePrimeList(1000)
for i in prime_list :
    if i in standard :
        total += 1
print(total)
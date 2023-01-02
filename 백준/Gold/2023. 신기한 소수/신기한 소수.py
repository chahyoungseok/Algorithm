import copy, sys


def isPrime(number) :
    if number == 2 :
        return True
    elif number == 1 or number % 2 == 0 :
        return False

    for n in range(2, int(number ** 0.5) + 1) :
        if number % n == 0 :
            return False

    return True


N = int(sys.stdin.readline().strip())

real = [2, 3, 5, 7]
for _ in range(1, N) :
    temp = []
    for i in real :
        for prime in range(1, 10) :
            target = int(str(i) + str(prime))
            if isPrime(target) :
                temp.append(target)

    real = copy.deepcopy(temp)

real = sorted(real)
for i in real :
    print(i)
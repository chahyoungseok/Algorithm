import math


def isPrime(number) :
    if number == 2:
        return True
    elif number % 2 == 0 or number == 1:
        return False

    for i in range(3, int(math.sqrt(number)) + 1, 2) :
        if number % i == 0 :
            return False
    return True


start, end = map(int, input().split())
for i in range(start, end + 1) :
    if isPrime(i) :
        print(i)
import math, sys


def isPrime(number) :
    if number == 2:
        return True
    elif number % 2 == 0 or number == 1 :
        return False

    for i in range(3, int(math.sqrt(number)) + 1, 2) :
        if number % i == 0 :
            return False
    return True


T = int(input())
for _ in range(T) :
    data = int(sys.stdin.readline())
    if data == 4 :
        print("2 2")
    else :
        standard = data // 2
        if standard % 2 == 0 :
            standard -= 1
        for i in range(standard, 1, -2) :
            if isPrime(i) and isPrime(data - i) :
                print(str(i) + " " + str(data - i))
                break
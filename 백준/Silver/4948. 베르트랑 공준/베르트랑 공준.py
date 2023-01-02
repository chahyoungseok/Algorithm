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


arr = [0] * 246913
for i in range(2, 246913) :
    if isPrime(i) :
        arr[i] = 1

while True :
    data = int(sys.stdin.readline())
    if data == 0 :
        break

    print(sum(arr[data + 1 : data * 2 + 1]))
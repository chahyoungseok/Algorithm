from itertools import permutations

def is_prime(number):
    if number == 2:
        return True
    elif number == 1 or number == 0 :
        return False
    for i in range(2, number):
        if number % i == 0:
            return False
    return True

def solution(numbers):
    answer = 0
    prime_list = []
    for i in range(1, len(numbers) + 1):
        for data in permutations(numbers, i):
            prime_list.append(int("".join(data)))

    prime_list = list(set(prime_list))
    for prime in prime_list:
        if is_prime(prime):
            answer += 1

    return answer
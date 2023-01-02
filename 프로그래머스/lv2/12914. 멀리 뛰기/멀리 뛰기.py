def factorial(n) :
    result = 1
    for i in range(1, n + 1) :
        result *= i
    return result


def solution(n):
    answer = 0
    two, one, n_range = 0, 0, n // 2

    for i in range(n_range + 1) :
        two = i
        one = n - (two * 2)

        result = int(factorial(one + two))
        result //= factorial(one)
        result //= factorial(two)
        answer += result

    return answer % 1234567
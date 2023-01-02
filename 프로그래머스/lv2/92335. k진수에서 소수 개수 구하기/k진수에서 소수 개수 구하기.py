def isPrime(number) :
    if number == 1 :
        return False
    
    if number == 2 :
        return True

    for i in range(3, int(number ** 0.5) + 1, 2) :
        if number % i == 0 :
            return False

    return True


def solution(n, k):
    trans_num, answer = [], 0
    for i in range(0, 21):
        if k ** i > n:
            max_digit = i - 1
            break

    for i in range(max_digit, -1, -1):
        for j in range(1, k + 1):
            if k ** i * j > n:
                trans_num.append(str(j - 1))
                n -= (k ** i) * (j - 1)
                break

    number = []
    for sel in trans_num :
        if sel == '0' :
            if number :
                num = int(''.join(number))
                if isPrime(num) :
                    answer += 1
                number = []
        else :
            number.append(sel)

    if number and int(''.join(number)) != 0 :
        if isPrime(int(''.join(number))) :
            answer += 1
    return answer
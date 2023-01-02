def solution(n):
    answer = list(str(n))
    answer.reverse()
    answer = list(map(int, answer))
    return answer
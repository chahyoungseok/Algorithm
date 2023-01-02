def solution(n):
    answer = sorted(list(str(n)), reverse=True)
    return int("".join(answer))
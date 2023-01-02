def solution(citations):
    answer, i, j = 0, 0, 0
    citations_size = len(citations)
    citations.sort()

    for i in range(1, citations_size + 1):
        for j in range(citations_size):
            if citations[j] >= i:
                break
            elif citations_size - 1 == j :
                j = citations_size
        if citations_size - j >= i:
            answer = i
        else:
            break

    return answer
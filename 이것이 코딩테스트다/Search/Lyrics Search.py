from bisect import bisect_left, bisect_right

def count_by_range(a, left_value, right_value) :
    right_index = bisect_right(a, right_value)
    left_index = bisect_left(a, left_value)
    return right_index - left_index


array = [[] for _ in range(10001)]
reversed_array = [[] for _ in range(10001)]


def solution(words, queries):
    answer = []

    for word in words :
        array[len(word)].append(word)
        reversed_array[len(word)].append(word[::-1])

    for i in range(10001) :
        array[i].sort()
        reversed_array[i].sort()

    for query in queries :
        if query[0] == '?' :
            res = count_by_range(reversed_array[len(query)], query[::-1].replace('?','a'), query[::-1].replace('?','z'))
        else :
            res = count_by_range(array[len(query)], query.replace('?','a'), query.replace('?','z'))

        answer.append(res)

    return answer


words = ["frodo", "front", "frost", "frozen", "frame", "kakao"]
queries = ["fro??", "????o", "fr???", "fro???", "pro?"]
print(solution(words, queries))
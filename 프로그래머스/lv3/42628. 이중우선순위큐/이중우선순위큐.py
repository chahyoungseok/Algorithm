import heapq

def solution(operations):
    answer = []

    for operation in operations:
        oper, num = operation.split(" ")
        if oper == "I":
            heapq.heappush(answer, int(num))
        elif oper == "D" and answer:
            if int(num) > 0:
                answer.pop(len(answer) - 1)
            else:
                heapq.heappop(answer)

    if answer :
        return [max(answer), min(answer)]
    else :
        return [0,0]
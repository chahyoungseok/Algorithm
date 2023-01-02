def solution(arr1, arr2):
    answer = []

    for i in range(len(arr1)):
        temporary = []
        for j in range(len(arr2[0])) :
            result = 0
            for k in range(len(arr2)) :
                result += arr1[i][k] * arr2[k][j]
            temporary.append(result)
        answer.append(temporary)
    return answer
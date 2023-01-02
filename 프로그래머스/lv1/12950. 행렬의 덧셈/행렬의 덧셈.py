def solution(arr1, arr2):
    arr1_len = len(arr1)
    answer = [[] for _ in range(arr1_len)]
    for i in range(arr1_len) :
        for j in range(len(arr1[0])) :
            answer[i].append(arr1[i][j] + arr2[i][j])
    return answer
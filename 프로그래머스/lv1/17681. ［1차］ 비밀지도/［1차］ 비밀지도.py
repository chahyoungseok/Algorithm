def arrToboard(n, arr) :
    answer = [["#" for _ in range(n)] for _ in range(n)]
    for i in range(n) :
        bin_value = str(format(arr[i], 'b'))
        while n > len(bin_value) :
            bin_value = "0" + bin_value

        for j in range(n) :
            if bin_value[j] == "0" :
                answer[i][j] = " "
            else :
                answer[i][j] = "#"

    return answer


def solution(n, arr1, arr2):
    board = [["#" for _ in range(n)] for _ in range(n)]

    arr1, arr2 = arrToboard(n, arr1), arrToboard(n, arr2)
    
    for i in range(n):
        for j in range(n):
            if arr1[i][j] == " " and arr2[i][j] == " ":
                board[i][j] = " "
        board[i] = "".join(board[i])
    return board
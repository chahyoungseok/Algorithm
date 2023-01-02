T = int(input())
for i in range(T) :
    arr = [[1, 0], [0, 1]]
    N = int(input())
    for j in range(1, N) :
        arr.append([arr[j - 1][0] + arr[j][0], arr[j - 1][1] + arr[j][1]])

    print(str(arr[N][0]) + " " + str(arr[N][1]))
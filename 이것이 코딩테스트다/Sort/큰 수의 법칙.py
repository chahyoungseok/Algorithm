// N개 주어진 자연수들을 연속으로 더하되 K번 초과하지않고 총 M번 더했을 때의 최대값 (중복가능)

N, M, K = map(int,input().split())

intArr = list(map(int,input().split()))
intArr.sort()

max = intArr[N - 1]
submax = intArr[N - 2]

count = int(M/(K + 1))

sum = submax * count + max * (M - count)
print(sum)
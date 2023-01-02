// 1. N에서 1을 뺀다.
// 2. N을 K로 나눈다. (나누어 떨어질때만)
// 위의 1 ,2를 1이 될 때까지 수행하는 최소연산의 수는?

N, K = map(int, input().split())

count = 0

while True :
    remainder = (N // K) * K
    count += N - remainder
    N = remainder

    if N < K :
        break

    count += 1
    N //= K

count += N - 1

print(count)
import sys

T = int(sys.stdin.readline().strip())
for _ in range(T) :
    n = int(sys.stdin.readline().strip())
    max_value = 0
    dp = [[0, 0] for _ in range(n)]
    first_s, second_s = list(map(int, (sys.stdin.readline()).split())), list(map(int, (sys.stdin.readline()).split()))

    if n == 1 :
        print(max(first_s[0], second_s[0]))
        continue

    dp[0] = [first_s[0], second_s[0]]
    dp[1] = [first_s[1] + second_s[0], first_s[0] + second_s[1]]
    for i in range(2, n) :
        dp[i][0] = max(dp[i][0], first_s[i] + dp[i - 1][1], first_s[i] + dp[i - 2][0], first_s[i] + dp[i - 2][1])
        dp[i][1] = max(dp[i][1], second_s[i] + dp[i - 1][0], second_s[i] + dp[i - 2][0], second_s[i] + dp[i - 2][1])

    for values in dp :
        value = max(values)
        if value > max_value :
            max_value = value

    print(max_value)
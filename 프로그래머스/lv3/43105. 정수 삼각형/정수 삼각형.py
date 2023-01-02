def solution(triangle):
    height = len(triangle)
    dp = [[0 for _ in range(i + 1)] for i in range(height)]
    
    dp[0][0] = triangle[0][0]
    for i in range(1, height) :
        for j in range(i) :
            dp[i][j] = max(dp[i][j], dp[i - 1][j] + triangle[i][j])
            dp[i][j + 1] = max(dp[i][j + 1], dp[i - 1][j] + triangle[i][j + 1])
    
    return max(dp[height - 1])        
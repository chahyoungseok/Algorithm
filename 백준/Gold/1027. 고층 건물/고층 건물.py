import sys

N = int(sys.stdin.readline().strip())
heights = list(map(int, (sys.stdin.readline()).split()))

max_count = 0
for i in range(N) :
    count = 0
    s_x, s_y = i, heights[i]

    for j in range(i - 1, -1, -1) :
        m_x, m_y = j, heights[j]
        state = True
        for k in range(i - 1, j, -1) :
            a = (s_y - m_y) / (s_x - m_x)
            if heights[k] >= s_y - (i - k) * a:
                state = False
                break
        if state :
            count += 1
    for j in range(i + 1, N, 1) :
        m_x, m_y = j, heights[j]
        state = True
        for k in range(i + 1, j, 1):
            a = (s_y - m_y) / (s_x - m_x)
            if heights[k] >= s_y + (k - i) * a:
                state = False
                break
        if state:
            count += 1

    max_count = max(max_count, count)

print(max_count)
def solution(N, stages):
    stages_count = [0 for _ in range(N + 1)]
    failure = [[i, 0] for i in range(1, N + 1)]

    for i in stages:
        stages_count[i - 1] += 1

    remain = len(stages)
    for i in range(N):
        if remain <= 0:
            failure[i][1] = 0
            continue
        failure[i][1] = stages_count[i] / remain
        remain -= stages_count[i]

    result = sorted(failure, key=lambda x: (x[1], -x[0]), reverse=True)

    return [i[0] for i in result]
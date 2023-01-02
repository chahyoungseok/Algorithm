def solution(N, stages):
    failrate_stage = []
    stage_size = len(stages)

    for i in range(1, N + 1) :
        count = stages.count(i)

        if stage_size == 0 :
            failrate_stage.append((0,i))
        else :
            failrate_stage.append((count / stage_size, i))
            stage_size -= count

    failrate_stage.sort(key=lambda x : (-x[0], x[1]))

    return [i[1] for i in failrate_stage]

N = 5
stages = [2,1,2,6,2,4,3,3]
print(solution(N, stages))

N = 4
stages = [4,4,4,4,4]
print(solution(N, stages))
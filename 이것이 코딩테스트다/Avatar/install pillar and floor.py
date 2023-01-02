def possible(build) :
    for x, y, a in build :
        if a == 0 :  # 기둥
            if y == 0 or [x, y - 1, 0] in build or [x, y, 1] in build or [x - 1, y, 1] in build :
                continue
            return False
        else:  # 보
            if [x, y - 1, 0] in build or [x + 1, y - 1, 0] in build or ([x - 1, y, 1] in build and [x + 1, y, 1] in build) :
                continue
            return False
    return True

def solution(n, build_frame):
    build = []
    for frame in build_frame :
        x, y, a, b = frame
        if b == 0 :
            build.remove([x,y,a])
            if not possible(build) :
                build.append([x,y,a])
        else :
            build.append([x,y,a])
            if not possible(build) :
                build.remove([x,y,a])

    return sorted(build)

n = 5
build_frame = [[1,0,0,1],[1,1,1,1],[2,1,0,1],[2,2,1,1],[5,0,0,1],[5,1,0,1],[4,2,1,1],[3,2,1,1]]
# build_frame = [[0,0,0,1],[2,0,0,1],[4,0,0,1],[0,1,1,1],[1,1,1,1],[2,1,1,1],[3,1,1,1],[2,0,0,0],[1,1,1,0],[2,2,0,1]]
print(solution(n, build_frame))
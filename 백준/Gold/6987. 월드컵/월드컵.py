import copy, sys
from itertools import combinations


def dfs(copy_game, visited, standard):
    if standard == 6:
        total = 0
        for t in range(6):
            total += copy_game[t * 3 + 1] + copy_game[t * 3 + 2]
        if total == 0:
            global state
            state = True
            return "exit"
        else:
            return

    if copy_game[standard * 3] == 0:
        if copy_game[standard * 3 + 1] == 0:
            if dfs(copy_game, visited, standard + 1) == "exit":
                return "exit"
        else:
            for draw_comb in combinations(visited[standard], copy_game[standard * 3 + 1]):
                d_game, copy_draw, draw_state = copy.deepcopy(copy_game), copy.deepcopy(visited), True
                for d in draw_comb:
                    if d_game[d * 3 + 1] > 0 and d in copy_draw[standard]:
                        d_game[standard * 3 + 1] -= 1
                        d_game[d * 3 + 1] -= 1
                        copy_draw[standard].remove(d)
                        if standard in copy_draw[d] :
                            copy_draw[d].remove(standard)
                    else:
                        draw_state = False
                        break
                if draw_state:
                    if dfs(d_game, copy_draw, standard + 1) == "exit":
                        return "exit"
    else:
        for comb in combinations(visited[standard], copy_game[standard * 3]):
            c_game, copy_draw, win_loss_state = copy.deepcopy(copy_game), copy.deepcopy(visited), True
            for c in comb:
                if c_game[c * 3 + 2] > 0 and c in copy_draw[standard]:
                    c_game[c * 3 + 2] -= 1
                    copy_draw[standard].remove(c)
                    if standard in copy_draw[c] :
                        copy_draw[c].remove(standard)
                else:
                    win_loss_state = False
                    break

            if win_loss_state:
                draw_amount = copy_game[standard * 3 + 1]
                if draw_amount == 0:
                    if dfs(c_game, copy_draw, standard + 1) == "exit":
                        return "exit"
                elif len(copy_draw[standard]) >= draw_amount:
                    for draw_comb in combinations(copy_draw[standard], draw_amount):
                        d_c_game, d_c_draw, draw_state = copy.deepcopy(c_game), copy.deepcopy(copy_draw), True
                        for d in draw_comb :
                            if d_c_game[d * 3 + 1] > 0 and d in d_c_draw[standard]:
                                d_c_game[standard * 3 + 1] -= 1
                                d_c_game[d * 3 + 1] -= 1
                                d_c_draw[standard].remove(d)
                                if standard in d_c_draw[d] :
                                    d_c_draw[d].remove(standard)
                            else:
                                draw_state = False
                        if draw_state:
                            if dfs(d_c_game, d_c_draw, standard + 1) == "exit":
                                return "exit"


result = []
for _ in range(4):
    game = list(map(int, (sys.stdin.readline()).split()))
    state, sum_state, visited = False, True, [list(range(6)) for _ in range(6)]
    for i in range(6) :
        visited[i].remove(i)
    for s in range(6):
        if sum(game[(s * 3): (s * 3) + 3]) != 5:
            sum_state = False
            break

    if sum_state :
        dfs(game, visited, 0)

    if state:
        result.append(1)
    else:
        result.append(0)

for i in result:
    print(i, end=" ")

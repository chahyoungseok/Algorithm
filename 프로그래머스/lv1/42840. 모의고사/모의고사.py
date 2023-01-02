def solution(answers):
    answer = []
    give_up = [0,0,0]
    second_rule = [1, 3, 4, 5]
    third_rule = [3, 1, 2, 4, 5]

    for i in range(len(answers)) :
        if answers[i] == (i % 5) + 1 :
            give_up[0] += 1

        if i % 2 == 0:
            if answers[i] == 2:
                give_up[1] += 1
        else:
            if answers[i] == second_rule[(i // 2) % 4]:
                give_up[1] += 1

        if answers[i] == third_rule[(i // 2) % 5]:
            give_up[2] += 1

    for i in range(3) :
        if give_up[i] == max(give_up) :
            answer.append(i + 1)

    return answer
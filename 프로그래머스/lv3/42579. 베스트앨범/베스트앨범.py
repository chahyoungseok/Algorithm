def solution(genres, plays):
    answer, genres_len = [], len(genres)
    genres_dict, genres_sum = {}, {}

    for i in range(genres_len):
        if genres[i] in genres_dict.keys() :
            genres_dict[genres[i]].append([plays[i], i])
            genres_sum[genres[i]] += plays[i]
        else :
            genres_dict[genres[i]] = [[plays[i], i]]
            genres_sum[genres[i]] = plays[i]

    for i in sorted(genres_sum, key=lambda x : genres_sum[x], reverse=True) :
        dict_case = sorted(genres_dict[i], key=lambda x : -x[0])
        answer.append(dict_case[0][1])
        if len(dict_case) >= 2:
            answer.append(dict_case[1][1])

    return answer
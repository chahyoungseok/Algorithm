import re


def solution(word, pages):
    word = word.lower()
    site_dict, answer, answer_value, index = {}, 0, 0, 0
    for page in pages:
        url = re.search('<meta property="og:url" content="(.+)"', page).group(1)
        bs = 0
        for f in re.findall('[a-zA-Z]+', page.lower()) :
            if f == word :
                bs += 1

        el = re.findall('<a href="(\S+)"', page)

        site_dict[url] = [bs, el, 0]

    for key in site_dict.keys() :
        targets = site_dict[key][1]
        if len(site_dict[key][1]) != 0:
            link_score = site_dict[key][0] / len(site_dict[key][1])
            for target in targets :
                if target in site_dict.keys() :
                    site_dict[target][2] += link_score

    for key in site_dict.keys() :
        final_value = site_dict[key][0] + site_dict[key][2]
        if final_value > answer_value :
            answer = index
            answer_value = final_value
        index += 1

    return answer
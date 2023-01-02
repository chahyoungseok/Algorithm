start, end = map(int, input().split())

number_dict = {}
number_dict[start] = 0


def po(number, dist) :
    if number > 10 ** 9 :
        return
    number_dict[number] = dist
    po(number * 2, dist + 1)
    po(int(str(number) + '1'), dist + 1)

po(start, 1)
if end in number_dict.keys() :
    print(number_dict[end])
else :
    print(-1)

A, B = map(int, input().split())
easy_list = []
index = 1
while 1000 >= len(easy_list) :
    for _ in range(index) :
        easy_list.append(index)
    index += 1

print(sum(easy_list[A - 1 : B]))
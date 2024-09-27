import sys


def valid_install(min_distance):
    install_count = 1
    pre_house = house_list[0]

    for house in house_list :
        if house - pre_house >= min_distance :
            install_count += 1
            pre_house = house

    if install_count >= C :
        return True
    else :
        return False

def binary_search(start, end):
    mid = (start + end) // 2

    if start > end:
        return mid

    if valid_install(mid):
        return binary_search(mid + 1, end)
    else :
        return binary_search(start, mid - 1)

N, C = map(int, (sys.stdin.readline().strip()).split())

house_list = []
for _ in range(N):
    house_list.append(int(sys.stdin.readline().strip()))

house_list = sorted(house_list)

print(binary_search(1, max(house_list) // (C - 1)))
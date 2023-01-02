N = int(input())
house_list = list(map(int, input().split()))
house_list.sort()

print(house_list[(N-1) // 2])
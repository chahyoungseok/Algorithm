import sys

H, W = map(int, input().split())
height = list(map(int, (sys.stdin.readline()).split()))
total = 0

max_index = 0
for i in range(1, W) :
    if height[i] > height[max_index] :
        max_index = i

left_point, right_point = max_index, max_index

while left_point > 1 :
    max_index = 0
    for i in range(1, left_point, 1):
        if height[i] >= height[max_index]:
            max_index = i

    for i in range(max_index + 1, left_point) :
        total += height[max_index] - height[i]

    left_point = max_index

while right_point < W - 2:
    max_index = W - 1
    for i in range(W - 2, right_point, -1):
        if height[i] >= height[max_index]:
            max_index = i

    for i in range(right_point + 1, max_index):
        total += height[max_index] - height[i]

    right_point = max_index

print(total)
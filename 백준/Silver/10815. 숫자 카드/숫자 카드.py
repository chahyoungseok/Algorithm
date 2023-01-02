import sys


def binary_search(arr, target, start, end) :
    if start > end :
        return -1
    mid = (start + end) // 2

    if arr[mid] > target :
        return binary_search(arr, target, start, mid - 1)
    elif arr[mid] < target :
        return binary_search(arr, target, mid + 1, end)
    else :
        return mid


N = int(input())
number_cards = list(map(int, sys.stdin.readline().split()))
number_cards = sorted(number_cards)

M = int(input())
sang_cards = list(map(int, sys.stdin.readline().split()))


for card in sang_cards :
    if binary_search(number_cards, card, 0, N - 1) != -1 :
        print("1", end=" ")
    else :
        print("0", end=" ")
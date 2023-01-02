import sys
from bisect import bisect_left, bisect_right


N = int(input())

sang_cards = list(map(int, (sys.stdin.readline()).split()))
sang_cards = sorted(sang_cards)

M = int(input())
confirm_cards = list(map(int, (sys.stdin.readline()).split()))

for card in confirm_cards :
    total = 0
    print(bisect_right(sang_cards, card) - bisect_left(sang_cards, card), end=" ")


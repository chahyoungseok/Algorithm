def solution(stones, k):
    stones_len = len(stones)
    if sum(stones[0:stones_len // 2]) > sum(stones[stones_len // 2: stones_len]):
        stones.reverse()

    representative_stone = max(stones[0: k])
    min_count = representative_stone

    for i in range(1, stones_len - k + 1):
        if representative_stone == stones[i - 1]:
            if representative_stone > stones[i + k - 1]:
                representative_stone = max(stones[i: i + k])

            elif representative_stone < stones[i + k - 1]:
                representative_stone = stones[i + k - 1]
        else:
            if stones[i + k - 1] > representative_stone:
                representative_stone = stones[i + k - 1]
        if min_count > representative_stone:
            min_count = representative_stone

    return min_count
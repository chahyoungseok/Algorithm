import sys


def ipToInt(octet_list):
    result = 0
    for octet_index in range(4):
        result |= (octet_list[octet_index] << 24 - (octet_index * 8))

    return result


def intToIp(value):
    octet_standard = (1 << 8) - 1
    return f"{(value >> 24) & octet_standard}.{(value >> 16) & octet_standard}.{(value >> 8) & octet_standard}.{value & octet_standard}"


def findMask(min_value, max_value):
    mask = 0xFFFFFFFF
    while True:
        network_min = min_value & mask
        network_max = network_min | ~mask & 0xFFFFFFFF

        # max_value가 네트워크 범위 안에 들어오면 반복 종료
        if network_min <= max_value <= network_max:
            break

        mask <<= 1

    return mask & 0xFFFFFFFF


N = int(sys.stdin.readline().strip())
min_value, max_value = sys.maxsize, 0
for _ in range(N):
    octets = list(map(int, sys.stdin.readline().strip().split(".")))

    value = ipToInt(octets)
    min_value, max_value = min(min_value, value), max(max_value, value)

subnet_mask = findMask(min_value, max_value)
network_address = min_value & subnet_mask

print(intToIp(network_address))
print(intToIp(subnet_mask))

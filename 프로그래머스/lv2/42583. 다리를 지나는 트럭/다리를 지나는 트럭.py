from collections import deque

def solution(bridge_length, weight, truck_weights):
    answer, truck_sum = 0, 0
    bridge = deque([])

    while truck_weights or bridge :

        for truck in bridge:
            truck[1] += 1

        if bridge and bridge[0][1] >= bridge_length :
            escape_truck = bridge.popleft()
            truck_sum -= escape_truck[0]

        if truck_weights and bridge_length > len(bridge) and weight >= truck_sum + truck_weights[0]:
            in_bridge = truck_weights.pop(0)
            bridge.append([in_bridge, 0])
            truck_sum += in_bridge

        answer += 1

    return answer
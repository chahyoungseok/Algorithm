import sys


def find_gate(x):
    if gates[x][0] != x:
        gates[x][0] = find_gate(gates[x][0])
    return gates[x][0]


def used_gate(x):
    gates[x][1] = False

    if 1 > x - 1:
        return

    gates[x][0] = find_gate(x - 1)


G = int(sys.stdin.readline().strip())
P = int(sys.stdin.readline().strip())

result = 0
gates = [[ele, True] for ele in range(G + 1)]
for count in range(P):
    result += 1
    gi = int(sys.stdin.readline().strip())
    if gates[gi][1]:
        used_gate(gi)
    else:
        if 1 > gi - 1:
            result -= 1
            break

        parent_gate = find_gate(gi - 1)
        if not gates[parent_gate][1]:
            result -= 1
            break

        used_gate(parent_gate)
        gates[gi][0] = find_gate(parent_gate)


print(result)
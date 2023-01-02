import sys

N, M = map(int, (sys.stdin.readline()).split())
packages, costs = [], []
for _ in range(M) :
    package, cost = map(int, (sys.stdin.readline()).split())
    packages.append(package)
    costs.append(cost)

package_min = min(min(packages), min(costs) * 6)
if package_min > (N % 6) * min(costs) :
    print(package_min * (N // 6) + (N % 6) * min(costs))
else :
    print(package_min * (N // 6) + package_min)
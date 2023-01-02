def gcd(n,m):
    mod = m % n
    if mod != 0:
        m, n = n, mod
        return gcd(n, m)
    else:
        return n


def solution(n,m):
    gcd_nm = gcd(n,m)
    return [gcd_nm,int(m*n/gcd_nm)]
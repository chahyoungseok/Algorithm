def solution(s) :
    s_len = len(s)
    if s_len <= 1 :
        return s_len

    minLen = int(1e9)

    for length in range(1, s_len // 2 + 1):
        compressStr = ""
        sameCount = 1
        preStr = s[0:length]

        for i in range(length,s_len + 1,length) :
            if s_len >= i + length :
                if preStr == s[i: i + length]:
                    sameCount += 1
                else:
                    if sameCount != 1 :
                        compressStr += str(sameCount) + preStr
                    else :
                        compressStr += preStr

                    preStr = s[i: i + length]
                    sameCount = 1
            else :
                if sameCount != 1:
                    compressStr += str(sameCount) + preStr
                else :
                    compressStr += preStr

                compressStr += s[i : s_len]

        minLen = min(minLen, len(compressStr))

    return minLen
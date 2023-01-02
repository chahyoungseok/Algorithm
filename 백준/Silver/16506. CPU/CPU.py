import sys

N = int(sys.stdin.readline().strip())

opc_dict = {}
opc_dict["ADD"], opc_dict["ADDC"] = "00000", "00001"
opc_dict["SUB"], opc_dict["SUBC"] = "00010", "00011"
opc_dict["MOV"], opc_dict["MOVC"] = "00100", "00101"
opc_dict["AND"], opc_dict["ANDC"] = "00110", "00111"
opc_dict["OR"], opc_dict["ORC"] = "01000", "01001"
opc_dict["NOT"] = "01010"
opc_dict["MULT"], opc_dict["MULTC"] = "01100", "01101"
opc_dict["LSFTL"], opc_dict["LSFTLC"] = "01110", "01111"
opc_dict["LSFTR"], opc_dict["LSFTRC"] = "10000", "10001"
opc_dict["ASFTR"], opc_dict["ASFTRC"] = "10010", "10011"
opc_dict["RL"], opc_dict["RLC"] = "10100", "10101"
opc_dict["RR"], opc_dict["RRC"] = "10110", "10111"

for _ in range(N) :
    result = ""
    opc, rD, rA, data = map(str, sys.stdin.readline().split())
    opc_2 = opc_dict[opc]
    result += opc_2 + "0"

    rD_2 = str(format(int(rD), 'b'))
    for _ in range(3 - len(rD_2)) :
        rD_2 = "0" + rD_2
    result += rD_2

    rA_2 = str(format(int(rA), 'b'))
    for _ in range(3 - len(rA_2)):
        rA_2 = "0" + rA_2
    result += rA_2

    if opc_2[4] == "0" :
        data_2 = str(format(int(data), 'b'))
        for _ in range(3 - len(data_2)):
            data_2 = "0" + data_2
        result += data_2
        result += "0"
    else :
        data_2 = str(format(int(data), 'b'))
        for _ in range(4 - len(data_2)):
            data_2 = "0" + data_2
        result += data_2

    print(result)
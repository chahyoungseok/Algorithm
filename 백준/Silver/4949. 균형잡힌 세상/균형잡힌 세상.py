def is_balance(data) :
    while ("()" in data) or ("[]" in data) :
        if "()" in data :
            data = data.replace("()", "")
        if "[]" in data :
            data = data.replace("[]", "")

    if data == "" :
        return "yes"
    else:
        return "no"


while True :
    data = input()
    if data == '.':
        break
    real_data = ""
    for i in data :
        if i == "(" or i == ")" or i == "[" or i == "]" :
            real_data += i

    print(is_balance(real_data))
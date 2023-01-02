def solution(new_id):
    new_id = new_id.lower()

    for i in list("~!@#$%^&*()=+[{]}:?,<>/"):
        new_id = new_id.replace(i, "")

    new_id_size, new_id = len(new_id), list(new_id)

    for i in range(new_id_size):
        if new_id[i] == "." and new_id_size > i + 1 and new_id[i + 1] == ".":
            new_id[i] = ""

    new_id = list("".join(new_id))

    if new_id[0] == ".":
        new_id[0] = ""

    new_id = list("".join(new_id))

    if len(new_id) != 0 and new_id[len(new_id) - 1] == ".":
        new_id[len(new_id) - 1] = ""

    new_id = list("".join(new_id))

    if not new_id:
        new_id = "a"

    if len(new_id) >= 16:
        new_id = new_id[:15]
        if new_id[14] == ".":
            new_id_list = list(new_id)
            new_id_list[14] = ""
            new_id = "".join(new_id_list)

    if 2 >= len(new_id):
        standard = new_id[len(new_id) - 1]
        while len(new_id) != 3:
            new_id += standard

    return "".join(new_id)
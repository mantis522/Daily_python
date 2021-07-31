def solution(array, commands):
    return_list = []
    for a in range(len(commands)):
        new_commands = commands[a]
        sliced_array = array[new_commands[0]-1:new_commands[1]]
        sliced_array.sort()
        num = sliced_array[new_commands[2]-1]
        return_list.append(num)

    return return_list


a = solution([1, 5, 2, 6, 3, 7, 4], [[2, 5, 3], [4, 4, 1], [1, 7, 3]])
print(a)
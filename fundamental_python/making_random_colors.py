from random import *

random_list = []

for i in range(106):
    (a, b, c) = randint(0, 256), randint(0, 256), randint(0, 256)
    random_list.append((a, b, c))

print(random_list)
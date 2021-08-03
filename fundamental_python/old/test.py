u = [[[], 1, []], 2, [[[], 3, []], 4, [[], 5, []]]]
v = [[], 1, []]

# def depth(t):
#     if t == []:
#         return 0
#     else:


def add(x, t):
    if t == []:
        return [[], x, []]
    else:
        [l, y, r] = t
        if x == y:
            return t
        elif x < y:
            return [add(x, l), y, r]
        else:
            return [l, y, add(x, r)]


def succ(A, x):
    M = []
    Q = [x]
    while Q != []:
       u = Q.pop()
       M.append(u)
       for v in A[u]:
           if not v in M:
               Q.append(v)
    return M

A = {1:[1, 2, 5], 2:[6], 3:[2], 4:[5], 5:[2,6], 6:[3]}
print(succ(A, 4))

def cyclic(A, x):
    M = []
    Q = [x]
    while Q != []:
       u = Q.pop()
       M.append(u)
       for v in A[u]:
           if not v in M:
               Q.append(v)

    for a in range(len(M)-1):
        M.pop()

    print(M)

    # if x in M:
    #     return True
    # elif not x in M:
    #     return False

print(cyclic(A, 5))


def bfs(graph, start_node):
    visit = []
    queue = []

    queue.append(start_node)

    while queue:
        node = queue.pop(0)
        if node not in visit:
            visit.append(node)
            queue.extend(graph[node])

    return visit

# print(bfs(A, 4))


def cycle_exists(G):  # - G is a directed graph
    color = {u: "white" for u in G}  # - All nodes are initially white
    found_cycle = [False]  # - Define found_cycle as a list so we can change
    # its value per reference, see:
    # http://stackoverflow.com/questions/11222440/python-variable-reference-assignment
    for u in G:  # - Visit all nodes.
        if color[u] == "white":
            dfs_visit(G, u, color, found_cycle)
        if found_cycle[0]:
            break
    return found_cycle[0]


# -------

def dfs_visit(G, u, color, found_cycle):
    if found_cycle[0]:  # - Stop dfs if cycle is found.
        return
    color[u] = "gray"  # - Gray nodes are in the current path
    for v in G[u]:  # - Check neighbors, where G[u] is the adjacency list of u.
        if color[v] == "gray":  # - Case where a loop in the current path is present.
            found_cycle[0] = True
            return
        if color[v] == "white":  # - Call dfs_visit recursively.
            dfs_visit(G, v, color, found_cycle)
    color[u] = "black"  # - Mark node as done.

print(cycle_exists(A))
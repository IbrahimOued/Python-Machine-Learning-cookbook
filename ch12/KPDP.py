# 1 This algorithm starts with the definition of a KnapSackTable()
# function that will choose the optimal combination of the objects
# respecting the two constraints imposed by the problem: the total 
# weight of the objects equal to 10, and the maximum
# value of the chosen objects, as shown in the following code:
def KnapSackTable(weight, value, P, n):
    T = [[0 for w in range(P + 1)] for i in range(n+1)]

    # 2 Then we set an iterative loop on all objects and all weights valuesn
    for i in range(n+1):
        for w in range(P+1):
            if i == 0 or w == 0:
                T[i][w] = 0
            elif weight[i-1] <= w:
                T[i][w] = max(value[i-1] + T[i-1][w-weight[i-1]], T[i-1][w])
            else:
                T[i][w] = T[i-1][w]

        # 3 Now, we can memorize the result we have obtained, which represents
        # the maximum value of the objects that can be carried in the knapstack
        res = T[n][P]
        print("The maximum value of the objects that can be carried in the knapstack is:", res)

        # 4 The procedure we've folloxed so far does not indicate which subset provides
        # the optimal solution. We must extract this information using a set of procedure

        w = P
        totweight = 0
        for i in range(n, 0, -1):
            if res <= 0:
                break
        # 5 If the current element is the same as the previous one, we will
        # move to the next one
        if res == T[i-1][w]:
            continue
        # 6 If it is not the same, then the current object will be included in the
        # knapstack
        else:
            print('Item selected: ', weight[i-1], value[i-1])
            totweight += weight[i-1]
            res = res - value[i - 1]
            w = w - weight[i-1]
    # 7 Finally, the total included weight is print
    print("Total weight: ", totweight)
    # In this way, we have defined the function that allows us to build the table

# 8 Now we have to define the input variables and pass them to the function
objects = [(5, 18),(2, 9), (4, 12), (6,25)]
print("Items available: ",objects)
print("***********************************")

# 9 At this point we need to extract he weight and variable values from the objects
# we put them in a separate array to better understand the steps
value = []
weight = []
for item in objects:
    weight.append(item[0])
    value.append(item[1])

# 10 Finally the total weight that can be carried by the knapsack and the number
# of available items is set, as follows:
P = 10
n = len(value)

# 11 Finally, let's run and see the results    
KnapSackTable(weight, value, P, n)

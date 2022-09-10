# 1 Let's make the basic imports
import numpy as np
from functools import reduce

# 2 Let's define a function to add 3 to each element of the array
def add3(input_arry):
    return map(lambda x: x+3, input_arry)

# 3 Now let's define a second function to multiply 2 with each element of the array
def mul2(input_array): return map(lambda x: x*2, input_array)

# 4 Now let's define a 3rd function to substract 5 from each element of the array
def sub5(input_array): return map(lambda x: x-5, input_array)

# 5 Letdefine a function composer that takes functions as input arguments
# and returns a composed function. This composed
# function is basically a function that applies all the
# input functions in a sequence:
def function_composer(*args): 
    return reduce(lambda f, g: lambda x: f(g(x)), args)
    # We use the reduce function to combine all the input functions by
    # successively applying the functions in a sequence.

# 6 We are now ready to play with this function composer.
# Let's define some data and a sequence of operations
if __name__=='__main__': 
    arr = np.array([2,5,4,7]) 

    print("Operation: add3(mul2(sub5(arr)))")

    # 7 If we use the regular method, we apply this successively, as follows:
    arr1 = add3(arr) 
    arr2 = mul2(arr1) 
    arr3 = sub5(arr2) 
    print("Output using the lengthy way:", list(arr3))

    # 8 Now, let's use the function composer to achieve the same thing in a single line:
    func_composed = function_composer(sub5, mul2, add3) 
    print("Output using function composition:", list(func_composed(arr)))

    # 9 We can do the same thing in a single line with the previous method as well,
    # but the notation becomes very nested and unreadable. Also, it is not reusable; you will
    # have to write the whole thing again if you want to reuse this sequence of operations:
    print("Operation: sub5(add3(mul2(sub5(mul2(arr)))))\nOutput:", list(function_composer(mul2, sub5, mul2, add3, sub5)(arr)))

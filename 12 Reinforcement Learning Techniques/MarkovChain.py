# 1 Let's make the basic imports
import numpy as np
import time
from matplotlib import pyplot

# 2 Let's see the seed of a random generator
# and the state of the weather
np.random.seed(1)
states = ["Sunny", "Rainy"]

# 3 At this point, we have to define the possible
# transitions of weather conditions
TransStates = [["Susu", "SuRa"], ["RaRa", "RaSu"]]
TransnMatrix = [[.75, .25], [.30, .70]]

# 4 Then, we insert the following check to verify that we did not make mistakes in defining the transition matrix:
if sum(TransnMatrix[0])+sum(TransnMatrix[1]) != 2:
    print("Warning! Probabilities MUST ADD TO 1. Wrong transition matrix!!")
    raise ValueError("Probabilities MUST ADD TO 1")

# 5 Let's set the initial condition:
WT = list()
NumberDays = 200
WeatherToday = states[0]
print("Weather initial condition =", WeatherToday)

# 6 We can now predict the weather conditions for each of the days set by the NumberDays
# variable. To do this, we will use a while loop, as follows:
i = 0
while i < NumberDays:
    if WeatherToday == "Sunny":
        TransWeather = np.random.choice(
            TransStates[0], replace=True, p=TransnMatrix[0])
        if TransWeather == "SuSu":
            pass
        else:
            WeatherToday = "Rainy"
    elif WeatherToday == "Rainy":
        TransWeather = np.random.choice(
            TransStates[1], replace=True, p=TransnMatrix[1])
        if TransWeather == "RaRa":
            pass
        else:
            WeatherToday = "Sunny"
    print(WeatherToday)
    WT.append(WeatherToday)
    i += 1
    time.sleep(0.2)
# It consists of a control condition and a loop body.
# At the entrance of the cycle and every time that all the instructions
# contained in the body are executed, the validity of the control
# condition is verified. The cycle ends when the condition, consisting of a Boolean expression, returns false.

# 7 At this point, we have generated forecasts for the next 200 days. Let's plot the chart using the following code
pyplot.plot(WT)
pyplot.show()

# The following graph shows the weather conditions for the next 200 days, starting from the sunny condition
# At first sight, it seems that sunny days prevail over the rainy ones.

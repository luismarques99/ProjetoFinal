# from src.sprint_2_arima_automation import arima_automated
from src.sprint_3_arima_variations import arima_improved
# from src.sprint_3_arima_variations import arima_multivariate_improved

# arima_automated.init()
arima_improved.init()
# arima_multivariate_improved.init()

# from matplotlib import pyplot
# import numpy
#
# numbers = [6, 2, 3, 7, 8, 2, 1, 14, 19, 37]
# times = range(15)
# predictions = [32, 12, 25, 64, 23]
#
# pyplot.plot(range(len(numbers)), numbers, color="green", marker="8", label="numbers")
# pyplot.plot(times,
#             [None for i in numbers[:-1]] + [numbers[len(numbers) - 1]] + [x for x in predictions],
#             color="red",
#             marker="8",
#             label="numbers")
# pyplot.ylabel("numbers")
# pyplot.xlabel("timesteps")
# pyplot.xticks(numpy.arange(min(times), max(times) + 1, 1.0))
# pyplot.gcf().canvas.set_window_title("Numbers")
# pyplot.grid(alpha=0.5, color="#000000", linestyle=":")
# pyplot.gcf().set_size_inches(8, 6)
# pyplot.show()

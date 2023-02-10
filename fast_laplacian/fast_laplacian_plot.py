import matplotlib.pyplot as plt

old_times = []
new_times = []
num_ns = len(old_times)


ns = [i + 3 for i in range(num_ns)]
plt.plot(ns, new_times, label="New method")
plt.plot(ns, old_times, label="Old method")
plt.yscale("log")
plt.legend()
plt.show()

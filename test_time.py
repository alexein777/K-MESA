from clustering.utils import time_elapsed
import numpy as np
import time

x = 10

start = time.time_ns()
for i in range(500000):
    x *= 3
end = time.time_ns()

print(f'type(start) = {type(start)}')
print(f'end - start = {end - start}')
print(f'{(end - start) / 1000000}')

time_string = time_elapsed(start, end)
print(f'Time elapsed: {time_string}')
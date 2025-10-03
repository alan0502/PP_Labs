# Convert all values to milliseconds, then sum them
values_ms = [
    284.523,        # ms
    1.156 * 1000,   # s -> ms
    4.327,          # ms
    6.443,          # ms
    280.170,        # ms
    1.068           # ms
]

total_ms = sum(values_ms)
total_s = total_ms / 1000  # convert to seconds

print(total_ms, total_s)
io_times   = [1.14, 1.77, 1.66]   # Case1, Case2, Case3
comm_times = [5.46, 2.80, 2.73]
comp_times = [4.92, 4.43, 4.50]
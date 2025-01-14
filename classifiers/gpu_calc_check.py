import cuml
import cuml.common.logger as logger

def check_gpu():
    if cuml.has_cuML():
        print("cuML is available.")
        gpu_info = logger.get_devices_info()
        print(gpu_info)
    else:
        print("cuML is not available.")

# Call the check_gpu function
check_gpu()

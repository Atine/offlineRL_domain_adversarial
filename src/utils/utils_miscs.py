# common
import time
import subprocess


def auto_select_gpu(
    threshold_vram_usage=10, wait=False, sleep_time=10, force_select=False
):
    """
    from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Assigns free gpus to the current process via the
    CUDA_AVAILABLE_DEVICES env variable.
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the
                                              vram usage is below the
                                              threshold. Defaults to 10 MB.
        wait (bool, optional): Whether to wait until a GPU is free.
                               Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before
                                    checking GPUs, if wait=True. Default 10.
    """

    if type(force_select) == int:
        print(f"Selected GPU no: {force_select} to use by designation")
        print()
        return force_select

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )

        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))

        # Remove garbage
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]

        # Keep gpus under threshold only
        free_gpus = [
            str(i)
            for i, mem in enumerate(gpu_info)
            if mem < threshold_vram_usage
        ]

        if len(free_gpus) > 0:
            return True, max(free_gpus)
        else:
            return False, gpu_info.index(min(gpu_info))

    while True:
        success, gpus_to_use = _check()
        if success or not wait:
            break
        print(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not success:
        print("No free GPUs found.")
        print('Can be forced by designinating gpu="{}" as command argument')
        print("Exiting...")
        print()
        exit()

    print(f"Selected GPU no: {gpus_to_use} to use automatically")
    print()
    return gpus_to_use


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

import torch

class CudaTimer:
    def __init__(self, name: str | None = None, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start_event.record()

    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            self.end_event.synchronize()
            time_usage = self.start_event.elapsed_time(self.end_event)
            if self.name is not None:
                print(f"{self.name}: {time_usage:.2f} ms")
            else:
                print(f"{time_usage:.2f} ms")

class NvtxRange:
    def __init__(self, name: str):
        self.name = name
    
    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
    
    def __exit__(self, *args):
        torch.cuda.nvtx.range_pop()
        
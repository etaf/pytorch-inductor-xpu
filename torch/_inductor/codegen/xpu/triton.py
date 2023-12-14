from ..triton import TritonScheduling, DeviceApiCodeGen

class XPUDeviceApiCodeGen(DeviceApiCodeGen):
    def codegen_sync():
        V.graph.wrapper_code.writeline("torch.xpu.synchronize()")

    def codegen_import_get_raw_stream(self):
        return "from torch._C import _cuda_getCurrentRawStream as get_raw_stream"

    def codegen_device_guard(self, device_idx):
        return f"torch.xpu._DeviceGuard({device_idx})"

    def codegen_set_device(self, device_idx):
        return f"torch.xpu.set_device({device_idx})"


class XPUTritonScheduling(TritonScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self.device_api_codegen = XPUDeviceApiCodeGen()

    def codegen_sync(self):
        V.graph.wrapper_code.writeline("torch.xpu.synchronize()")

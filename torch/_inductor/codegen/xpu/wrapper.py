from ..wrapper import WrapperCodeGen, EnterDeviceContextManagerLine, ExitDeviceContextManagerLine

@dataclasses.dataclass
class EnterXPUDeviceContextManagerLine(EnterDeviceContextManagerLine):
    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if V.graph.cpp_wrapper:
            raise NotImplementedError
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with torch.xpu._DeviceGuard({self.device_idx}):")
            device_cm_stack.enter_context(code.indent())
            code.writeline(
                f"torch.xpu.set_device({self.device_idx}) # no-op to ensure context"
            )

@dataclasses.dataclass
class ExitXPUDeviceContextManagerLine(ExitDeviceContextManagerLine):
    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if not V.graph.cpp_wrapper:
            device_cm_stack.close()



class XPUTritonWrapperCodeGen(WrapperCodeGen):
    def __init__(self):
        super().__init__()

    def codegen_device_guard_enter(self, device_idx):
        self.writeline(
            EnterXPUDeviceContextManagerLine(device_idx, self.first_device_guard)
        )
        self.last_seen_device_guard_index = device_idx

    def codegen_device_guard_exit(self):
        self.writeline(ExitXPUDeviceContextManagerLine())

    def codegen_import_get_raw_stream(self):
        return "from torch._C import _xpu_getCurrentRawStream as get_raw_stream"

    def codegen_synchronize(self):
        return "torch.xpu.synchronize()"
 

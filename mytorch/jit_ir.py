# mytorch/jit_ir.py


class IRNode:
    """A node in the traced computation graph."""

    _counter = 0

    def __init__(self, op_name, inputs, output_shape, output_dtype=None, origin_func=None):
        self.op_name = op_name
        self.inputs = inputs
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.origin_func = origin_func
        self.constant_data = None
        self.optimization = None

        self.name = f"%{op_name}_{IRNode._counter}"
        IRNode._counter += 1

    def __repr__(self):
        formatted_inputs = []
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                formatted_inputs.append(inp.name)
            elif hasattr(inp, "shape"):
                formatted_inputs.append(f"Tensor(shape={inp.shape()})")
            else:
                formatted_inputs.append(str(inp))

        inputs_str = ", ".join(formatted_inputs)
        suffix = ""
        if self.constant_data is not None:
            suffix = " constant"
        elif self.optimization:
            suffix = f" {self.optimization}"
        return f"{self.name} = {self.op_name}({inputs_str})  # shape: {self.output_shape}{suffix}"


class IRGraph:
    """A traced computation graph."""

    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []

    def add_node(self, node):
        self.nodes.append(node)

    def set_outputs(self, outputs):
        self.outputs = list(outputs or [])

    def __repr__(self):
        graph_str = "graph(\n"
        for node in self.nodes:
            graph_str += f"  {node}\n"
        if self.outputs:
            graph_str += "  return "
            graph_str += ", ".join(node.name for node in self.outputs)
            graph_str += "\n"
        graph_str += ")"
        return graph_str


class TracerState:
    """Global tracing state."""

    _is_tracing = False
    _current_graph = None

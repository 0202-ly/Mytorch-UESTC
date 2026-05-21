# mytorch/jit_pass.py

from .jit_ir import IRNode, IRGraph


def count_consumers(graph: IRGraph):
    counts = {}
    for node in graph.nodes:
        for inp in node.inputs:
            if isinstance(inp, IRNode):
                counts[inp] = counts.get(inp, 0) + 1
    return counts


def _shape_of(value):
    if isinstance(value, IRNode):
        return value.output_shape
    if hasattr(value, "shape"):
        return value.shape() if callable(value.shape) else value.shape
    return None


def _dtype_of(value):
    if isinstance(value, IRNode):
        return value.output_dtype
    if hasattr(value, "dtype"):
        return str(value.dtype() if callable(value.dtype) else value.dtype)
    return None


def _clone_node(node, inputs):
    cloned = IRNode(
        node.op_name,
        inputs,
        node.output_shape,
        node.output_dtype,
        origin_func=node.origin_func,
    )
    cloned.name = node.name
    cloned.constant_data = getattr(node, "constant_data", None)
    cloned.optimization = getattr(node, "optimization", None)
    cloned.debug_data = getattr(node, "debug_data", None)
    cloned.is_train = getattr(node, "is_train", None)
    cloned.momentum = getattr(node, "momentum", None)
    cloned.eps = getattr(node, "eps", None)
    return cloned


def _is_tensor(value):
    return hasattr(value, "data") and hasattr(value, "requires_grad")


def _is_const_tensor(value):
    return _is_tensor(value) and not value.requires_grad


def _is_zero_const(value):
    if not _is_const_tensor(value):
        return False
    xp = value.xp
    try:
        return bool(xp.all(value.data == 0).item())
    except AttributeError:
        return bool(xp.all(value.data == 0))


def _can_fold_node(node):
    if node.op_name.startswith("Input"):
        return False
    if node.origin_func is None or getattr(node.origin_func, "data", None) is None:
        return False
    for inp in node.inputs:
        if isinstance(inp, IRNode):
            return False
        if _is_tensor(inp) and inp.requires_grad:
            return False
    return node.op_name in {
        "Add",
        "MatMul",
        "MSE",
        "Sum",
        "ReshapeOp",
        "ReLU",
        "ELU",
        "Sigmoid",
        "LogSoftmaxOp",
        "NLLLossOp",
    }


def _make_constant_from(node):
    const_node = IRNode(
        "Constant",
        [],
        node.output_shape,
        node.output_dtype,
        origin_func=None,
    )
    const_node.name = node.name
    const_node.constant_data = node.origin_func.data
    const_node.optimization = "constant-folded"
    return const_node


def _map_inputs(inputs, node_map):
    mapped = []
    for inp in inputs:
        if isinstance(inp, IRNode) and inp in node_map:
            mapped.append(node_map[inp])
        else:
            mapped.append(inp)
    return mapped


def can_fuse_bn_relu(bn_node, allow_training=False):
    if bn_node.op_name != "BatchNorm2dOp":
        return False
    if len(bn_node.inputs) < 5 or bn_node.origin_func is None:
        return False
    if getattr(bn_node.origin_func, "is_train", True) and not allow_training:
        return False

    gamma, beta, running_mean, running_var = bn_node.inputs[1:5]
    for tensor in (gamma, beta, running_mean, running_var):
        if _dtype_of(tensor) != "float32":
            return False
    return True


def can_fuse_conv_relu(conv_node):
    if conv_node.op_name != "Conv2dOp" or len(conv_node.inputs) < 7:
        return False

    x, w, b, stride, padding, dilation, groups = conv_node.inputs[:7]
    if b is None or groups <= 0:
        return False

    in_shape = _shape_of(x)
    weight_shape = _shape_of(w)
    if in_shape is None or weight_shape is None:
        return False

    in_c = in_shape[1]
    out_c = weight_shape[0]
    weight_in_c_per_group = weight_shape[1]

    if in_c % groups != 0 or out_c % groups != 0:
        return False
    if weight_in_c_per_group != in_c // groups:
        return False
    if _dtype_of(w) != "float32" or _dtype_of(b) != "float32":
        return False
    return True


def can_fuse_conv_bn(conv_node, bn_node, allow_training=False):
    if not can_fuse_conv_relu(conv_node):
        return False
    if bn_node.op_name != "BatchNorm2dOp" or len(bn_node.inputs) < 5:
        return False
    if bn_node.inputs[0] is not conv_node:
        return False
    if bn_node.origin_func is None:
        return False
    if getattr(bn_node.origin_func, "is_train", True) and not allow_training:
        return False

    gamma, beta, running_mean, running_var = bn_node.inputs[1:5]
    for tensor in (gamma, beta, running_mean, running_var):
        if _dtype_of(tensor) != "float32":
            return False
    return True


def _is_stem_conv(conv_node):
    if conv_node.op_name != "Conv2dOp" or not conv_node.inputs:
        return False
    x_shape = _shape_of(conv_node.inputs[0])
    if len(x_shape or ()) != 4:
        return False
    return x_shape[1] == 3


def can_fuse_linear_relu(matmul_node, add_node):
    if matmul_node.op_name != "MatMul" or len(matmul_node.inputs) != 2:
        return False
    if add_node.op_name != "Add" or len(add_node.inputs) != 2:
        return False

    x, weight = matmul_node.inputs
    if len(_shape_of(x) or ()) != 2 or len(_shape_of(weight) or ()) != 2:
        return False
    if _dtype_of(x) != "float32" or _dtype_of(weight) != "float32":
        return False
    return True


def _split_linear_add(add_node):
    left, right = add_node.inputs
    if isinstance(left, IRNode) and left.op_name == "MatMul":
        return left, right
    if isinstance(right, IRNode) and right.op_name == "MatMul":
        return right, left
    return None, None


def can_fuse_add_relu(add_node, relu_node):
    if add_node.op_name != "Add" or len(add_node.inputs) != 2:
        return False
    a, b = add_node.inputs
    if not isinstance(a, IRNode) or not isinstance(b, IRNode):
        return False
    # Keep this conservative: exact same-shape Add + ReLU, e.g. residual add.
    return _shape_of(a) == relu_node.output_shape and _shape_of(b) == relu_node.output_shape


def _prune_to_outputs(graph):
    outputs = list(graph.outputs)
    if not outputs and graph.nodes:
        outputs = [graph.nodes[-1]]
    if not outputs:
        return graph

    live = set()

    def visit(node):
        if not isinstance(node, IRNode) or node in live:
            return
        live.add(node)
        for inp in node.inputs:
            visit(inp)

    for output in outputs:
        visit(output)

    pruned = IRGraph()
    for node in graph.nodes:
        if node in live:
            pruned.add_node(node)
    pruned.set_outputs([node for node in outputs if node in live])
    return pruned


def optimize_graph(
    graph: IRGraph,
    training=False,
    disable_stem_fusion=False,
    disable_conv_bn_fusion=False,
) -> IRGraph:
    optimized_graph = IRGraph()
    node_map = {}
    consumer_count = count_consumers(graph)

    fused_counter = 0
    stem_conv_bn_relu_skipped = False

    for node in graph.nodes:
        new_inputs = _map_inputs(node.inputs, node_map)

        if _can_fold_node(node):
            const_node = _make_constant_from(node)
            optimized_graph.add_node(const_node)
            node_map[node] = const_node
            continue

        if node.op_name == "Add" and len(node.inputs) == 2:
            left, right = new_inputs
            if _is_zero_const(left) and _shape_of(right) == node.output_shape:
                node_map[node] = right
                continue
            if _is_zero_const(right) and _shape_of(left) == node.output_shape:
                node_map[node] = left
                continue

        if (
            node.op_name == "ReshapeOp"
            and len(new_inputs) >= 1
            and _shape_of(new_inputs[0]) == node.output_shape
        ):
            node_map[node] = new_inputs[0]
            continue

        if (
            node.op_name == "ReLU"
            and len(node.inputs) >= 1
            and isinstance(node.inputs[0], IRNode)
            and node.inputs[0].op_name == "ReLU"
        ):
            node_map[node] = new_inputs[0]
            continue

        if (
            node.op_name == "ReLU"
            and len(node.inputs) >= 1
            and isinstance(node.inputs[0], IRNode)
            and node.inputs[0].op_name == "BatchNorm2dOp"
        ):
            original_bn = node.inputs[0]
            original_conv = original_bn.inputs[0] if original_bn.inputs else None
            if (
                isinstance(original_conv, IRNode)
                and consumer_count.get(original_bn, 0) == 1
                and consumer_count.get(original_conv, 0) == 1
                and not disable_conv_bn_fusion
                and can_fuse_conv_bn(original_conv, original_bn, allow_training=training)
            ):
                skip_stem_fusion = (
                    disable_stem_fusion
                    and not stem_conv_bn_relu_skipped
                    and _is_stem_conv(original_conv)
                )
                if skip_stem_fusion:
                    stem_conv_bn_relu_skipped = True
                else:
                    conv_x = original_conv.inputs[0]
                    if isinstance(conv_x, IRNode) and conv_x in node_map:
                        conv_x = node_map[conv_x]

                    fused_node = IRNode(
                        op_name="FusedConvBNReLU",
                        inputs=[
                            conv_x,
                            original_conv.inputs[1],
                            original_conv.inputs[2],
                            original_bn.inputs[1],
                            original_bn.inputs[2],
                            original_bn.inputs[3],
                            original_bn.inputs[4],
                            original_conv.inputs[3],
                            original_conv.inputs[4],
                            original_conv.inputs[5],
                            original_conv.inputs[6],
                        ],
                        output_shape=node.output_shape,
                        output_dtype=node.output_dtype,
                        origin_func=None,
                    )
                    fused_node.name = f"%FusedConvBNReLU_{fused_counter}"
                    fused_node.optimization = "fused Conv2dOp+BatchNorm2dOp+ReLU via GEMM"
                    fused_node.is_train = getattr(original_bn.origin_func, "is_train", True)
                    fused_node.momentum = getattr(original_bn.origin_func, "momentum", 0.1)
                    fused_node.eps = getattr(original_bn.origin_func, "eps", 1e-5)
                    fused_counter += 1

                    for cloned in (node_map.get(original_conv), node_map.get(original_bn)):
                        if cloned in optimized_graph.nodes:
                            optimized_graph.nodes.remove(cloned)

                    optimized_graph.add_node(fused_node)
                    node_map[original_conv] = fused_node
                    node_map[original_bn] = fused_node
                    node_map[node] = fused_node
                    continue

            if (
                consumer_count.get(original_bn, 0) == 1
                and can_fuse_bn_relu(original_bn, allow_training=training)
            ):
                bn_x = original_bn.inputs[0]
                if isinstance(bn_x, IRNode) and bn_x in node_map:
                    bn_x = node_map[bn_x]

                fused_node = IRNode(
                    op_name="FusedBNReLU",
                    inputs=[
                        bn_x,
                        original_bn.inputs[1],
                        original_bn.inputs[2],
                        original_bn.inputs[3],
                        original_bn.inputs[4],
                        getattr(original_bn.origin_func, "eps", 1e-5),
                    ],
                    output_shape=node.output_shape,
                    output_dtype=node.output_dtype,
                    origin_func=None,
                )
                fused_node.name = f"%FusedBNReLU_{fused_counter}"
                fused_node.optimization = "fused BatchNorm2dOp+ReLU"
                fused_node.is_train = getattr(original_bn.origin_func, "is_train", True)
                fused_node.momentum = getattr(original_bn.origin_func, "momentum", 0.1)
                fused_node.eps = getattr(original_bn.origin_func, "eps", 1e-5)
                fused_counter += 1

                cloned_bn = node_map.get(original_bn)
                if cloned_bn in optimized_graph.nodes:
                    optimized_graph.nodes.remove(cloned_bn)

                optimized_graph.add_node(fused_node)
                node_map[original_bn] = fused_node
                node_map[node] = fused_node
                continue

        if (
            node.op_name == "ReLU"
            and len(node.inputs) >= 1
            and isinstance(node.inputs[0], IRNode)
            and node.inputs[0].op_name == "Conv2dOp"
        ):
            original_conv = node.inputs[0]
            if consumer_count.get(original_conv, 0) == 1 and can_fuse_conv_relu(original_conv):
                conv_x = original_conv.inputs[0]
                if isinstance(conv_x, IRNode) and conv_x in node_map:
                    conv_x = node_map[conv_x]

                fused_node = IRNode(
                    op_name="FusedConv2dReLU",
                    inputs=[
                        conv_x,
                        original_conv.inputs[1],
                        original_conv.inputs[2],
                        original_conv.inputs[3],
                        original_conv.inputs[4],
                        original_conv.inputs[5],
                        original_conv.inputs[6],
                    ],
                    output_shape=node.output_shape,
                    output_dtype=node.output_dtype,
                    origin_func=None,
                )
                fused_node.name = f"%FusedConv2dReLU_{fused_counter}"
                fused_node.optimization = "fused Conv2dOp+ReLU"
                fused_counter += 1

                cloned_conv = node_map.get(original_conv)
                if cloned_conv in optimized_graph.nodes:
                    optimized_graph.nodes.remove(cloned_conv)

                optimized_graph.add_node(fused_node)
                node_map[original_conv] = fused_node
                node_map[node] = fused_node
                continue

        if (
            node.op_name == "ReLU"
            and len(node.inputs) >= 1
            and isinstance(node.inputs[0], IRNode)
            and node.inputs[0].op_name == "Add"
        ):
            original_add = node.inputs[0]

            add_left, add_right = original_add.inputs
            original_bn = add_left if isinstance(add_left, IRNode) and add_left.op_name == "BatchNorm2dOp" else None
            identity = add_right
            if original_bn is None and isinstance(add_right, IRNode) and add_right.op_name == "BatchNorm2dOp":
                original_bn = add_right
                identity = add_left

            if original_bn is not None:
                original_conv = original_bn.inputs[0] if original_bn.inputs else None
                if (
                    isinstance(identity, IRNode)
                    and isinstance(original_conv, IRNode)
                    and consumer_count.get(original_add, 0) == 1
                    and consumer_count.get(original_bn, 0) == 1
                    and consumer_count.get(original_conv, 0) == 1
                    and not disable_conv_bn_fusion
                    and can_fuse_conv_bn(original_conv, original_bn, allow_training=training)
                    and _shape_of(identity) == node.output_shape
                ):
                    conv_x = original_conv.inputs[0]
                    if isinstance(conv_x, IRNode) and conv_x in node_map:
                        conv_x = node_map[conv_x]
                    mapped_identity = node_map.get(identity, identity)

                    fused_node = IRNode(
                        op_name="FusedConvBNAddReLU",
                        inputs=[
                            conv_x,
                            mapped_identity,
                            original_conv.inputs[1],
                            original_conv.inputs[2],
                            original_bn.inputs[1],
                            original_bn.inputs[2],
                            original_bn.inputs[3],
                            original_bn.inputs[4],
                            original_conv.inputs[3],
                            original_conv.inputs[4],
                            original_conv.inputs[5],
                            original_conv.inputs[6],
                        ],
                        output_shape=node.output_shape,
                        output_dtype=node.output_dtype,
                        origin_func=None,
                    )
                    fused_node.name = f"%FusedConvBNAddReLU_{fused_counter}"
                    fused_node.optimization = "fused Conv2dOp+BatchNorm2dOp+Add+ReLU via GEMM"
                    fused_node.is_train = getattr(original_bn.origin_func, "is_train", True)
                    fused_node.momentum = getattr(original_bn.origin_func, "momentum", 0.1)
                    fused_node.eps = getattr(original_bn.origin_func, "eps", 1e-5)
                    fused_counter += 1

                    for cloned in (
                        node_map.get(original_conv),
                        node_map.get(original_bn),
                        node_map.get(original_add),
                    ):
                        if cloned in optimized_graph.nodes:
                            optimized_graph.nodes.remove(cloned)

                    optimized_graph.add_node(fused_node)
                    node_map[original_conv] = fused_node
                    node_map[original_bn] = fused_node
                    node_map[original_add] = fused_node
                    node_map[node] = fused_node
                    continue

                if (
                    training
                    and isinstance(identity, IRNode)
                    and consumer_count.get(original_add, 0) == 1
                    and consumer_count.get(original_bn, 0) == 1
                    and can_fuse_bn_relu(original_bn, allow_training=training)
                    and _shape_of(identity) == node.output_shape
                ):
                    bn_x = original_bn.inputs[0]
                    if isinstance(bn_x, IRNode) and bn_x in node_map:
                        bn_x = node_map[bn_x]
                    mapped_identity = node_map.get(identity, identity)

                    fused_node = IRNode(
                        op_name="FusedBNAddReLU",
                        inputs=[
                            bn_x,
                            mapped_identity,
                            original_bn.inputs[1],
                            original_bn.inputs[2],
                            original_bn.inputs[3],
                            original_bn.inputs[4],
                            getattr(original_bn.origin_func, "eps", 1e-5),
                        ],
                        output_shape=node.output_shape,
                        output_dtype=node.output_dtype,
                        origin_func=None,
                    )
                    fused_node.name = f"%FusedBNAddReLU_{fused_counter}"
                    fused_node.optimization = "fused BatchNorm2dOp+Add+ReLU"
                    fused_node.is_train = getattr(original_bn.origin_func, "is_train", True)
                    fused_node.momentum = getattr(original_bn.origin_func, "momentum", 0.1)
                    fused_node.eps = getattr(original_bn.origin_func, "eps", 1e-5)
                    fused_counter += 1

                    for cloned in (node_map.get(original_bn), node_map.get(original_add)):
                        if cloned in optimized_graph.nodes:
                            optimized_graph.nodes.remove(cloned)

                    optimized_graph.add_node(fused_node)
                    node_map[original_bn] = fused_node
                    node_map[original_add] = fused_node
                    node_map[node] = fused_node
                    continue

            original_matmul, bias = _split_linear_add(original_add)
            if (
                isinstance(original_matmul, IRNode)
                and consumer_count.get(original_add, 0) == 1
                and consumer_count.get(original_matmul, 0) == 1
                and can_fuse_linear_relu(original_matmul, original_add)
                and not isinstance(bias, IRNode)
                and _dtype_of(bias) == "float32"
            ):
                x = original_matmul.inputs[0]
                if isinstance(x, IRNode) and x in node_map:
                    x = node_map[x]

                fused_node = IRNode(
                    op_name="FusedLinearReLU",
                    inputs=[
                        x,
                        original_matmul.inputs[1],
                        bias,
                    ],
                    output_shape=node.output_shape,
                    output_dtype=node.output_dtype,
                    origin_func=None,
                )
                fused_node.name = f"%FusedLinearReLU_{fused_counter}"
                fused_node.optimization = "fused MatMul+Add+ReLU"
                fused_counter += 1

                for cloned in (node_map.get(original_matmul), node_map.get(original_add)):
                    if cloned in optimized_graph.nodes:
                        optimized_graph.nodes.remove(cloned)

                optimized_graph.add_node(fused_node)
                node_map[original_matmul] = fused_node
                node_map[original_add] = fused_node
                node_map[node] = fused_node
                continue

            if consumer_count.get(original_add, 0) == 1 and can_fuse_add_relu(original_add, node):
                add_inputs = _map_inputs(original_add.inputs, node_map)
                fused_node = IRNode(
                    op_name="FusedAddReLU",
                    inputs=add_inputs,
                    output_shape=node.output_shape,
                    output_dtype=node.output_dtype,
                    origin_func=None,
                )
                fused_node.name = f"%FusedAddReLU_{fused_counter}"
                fused_node.optimization = "fused Add+ReLU"
                fused_counter += 1

                cloned_add = node_map.get(original_add)
                if cloned_add in optimized_graph.nodes:
                    optimized_graph.nodes.remove(cloned_add)

                optimized_graph.add_node(fused_node)
                node_map[original_add] = fused_node
                node_map[node] = fused_node
                continue

        new_node = _clone_node(node, new_inputs)
        optimized_graph.add_node(new_node)
        node_map[node] = new_node

    optimized_graph.set_outputs(_map_inputs(graph.outputs, node_map))
    return _prune_to_outputs(optimized_graph)

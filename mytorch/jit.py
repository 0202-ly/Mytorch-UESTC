# mytorch/jit.py
try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from operator import itemgetter
import time

from .jit_ir import TracerState, IRGraph, IRNode
from .tensor import Tensor
from .jit_pass import count_consumers, optimize_graph
from .jit_codegen import CUDACodegen
from .function import (
    Add,
    AvgPoolOp,
    BatchNorm2dOp,
    Conv2dOp,
    ConvTranspose2dOp,
    ELU,
    FusedAddReLUOp,
    FusedBatchNormAddReLUOp,
    FusedBatchNormReLUOp,
    FusedConv2dReLUOp,
    FusedConvBNAddReLUOp,
    FusedConvBNReLUOp,
    FusedCrossEntropyLossOp,
    FusedLinearReLUOp,
    FusedMSELossOp,
    LogSoftmaxOp,
    MatMul,
    MaxPoolOp,
    MinPoolOp,
    MSE,
    NLLLossOp,
    ReLU,
    ReshapeOp,
    Sigmoid,
    Sum,
)


def _collect_output_nodes(value):
    if isinstance(value, Tensor) and hasattr(value, "ir_node") and value.ir_node is not None:
        return [value.ir_node]
    if isinstance(value, (list, tuple)):
        nodes = []
        for item in value:
            nodes.extend(_collect_output_nodes(item))
        return nodes
    if isinstance(value, dict):
        nodes = []
        for item in value.values():
            nodes.extend(_collect_output_nodes(item))
        return nodes
    return []


def _pair(value):
    if isinstance(value, (tuple, list)):
        return value
    return value, value


def _prod(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def _iter_unique_tensors(obj, seen=None):
    if seen is None:
        seen = set()

    if isinstance(obj, Tensor):
        obj_id = id(obj)
        if obj_id not in seen:
            seen.add(obj_id)
            yield obj
        return

    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple, set)):
        values = obj
    elif hasattr(obj, "__dict__"):
        values = obj.__dict__.values()
    else:
        return

    for value in values:
        yield from _iter_unique_tensors(value, seen)


def _snapshot_model_tensors(model):
    snapshot = []
    for tensor in _iter_unique_tensors(model):
        snapshot.append((tensor, tensor.data.copy()))
    return snapshot


def _restore_model_tensors(snapshot):
    for tensor, data in snapshot:
        if tensor.data.shape == data.shape:
            tensor.data[...] = data
        else:
            tensor.data = data.copy()


class CompiledModule:
    """
    JIT wrapper.

    - training backend: trace the forward graph, optimize training-safe subgraphs,
      execute differentiable Function nodes, and leave backward to eager autograd.
    - inference backend: trace + optimize + execute CUDA RawKernel forward kernels.
    """

    def __init__(
        self,
        model,
        dump_graph=False,
        training=None,
        profile=False,
        disable_stem_fusion=False,
        disable_conv_bn_fusion=None,
        experimental_conv_bn_fusion=False,
    ):
        self.model = model
        self.cache = {}
        self.dump_graph = dump_graph
        # training=None means auto: model.train() uses differentiable graph execution,
        # model.eval() uses the CUDA RawKernel inference path.
        self.training_backend = training
        self.profile = profile
        self.disable_stem_fusion = disable_stem_fusion
        self.disable_conv_bn_fusion = disable_conv_bn_fusion
        self.experimental_conv_bn_fusion = experimental_conv_bn_fusion
        self.last_train_profile = None

    def _use_training_backend(self):
        if self.training_backend is None:
            return bool(getattr(self.model, "training", False))
        return bool(self.training_backend)

    def _disable_conv_bn_fusion_for(self, use_training_backend):
        if self.disable_conv_bn_fusion is not None:
            return bool(self.disable_conv_bn_fusion)
        if use_training_backend:
            return not bool(self.experimental_conv_bn_fusion)
        return False

    def __call__(self, *args):
        use_training_backend = self._use_training_backend()
        disable_conv_bn_fusion = self._disable_conv_bn_fusion_for(use_training_backend)
        if bool(getattr(self.model, "training", False)) and not use_training_backend:
            raise RuntimeError(
                "JIT inference RawKernel backend cannot be used while model.training=True, "
                "because it returns tensors without autograd creators. "
                "Use the training forward graph executor with jit.compile(model, training=True) "
                "or jit.compile_train(model), or call model.eval() before inference compilation."
            )

        cache_key = tuple(
            [("backend", "training" if use_training_backend else "inference"),
             ("model_training", bool(getattr(self.model, "training", False))),
             ("disable_stem_fusion", bool(self.disable_stem_fusion)),
             ("disable_conv_bn_fusion", bool(disable_conv_bn_fusion)),
             ("experimental_conv_bn_fusion", bool(self.experimental_conv_bn_fusion))]
            + [
            (arg.shape(), str(arg.dtype()), arg.device())
            for arg in args
            if isinstance(arg, Tensor)
            ]
        )
        if cache_key in self.cache:
            return self._execute_fast_path(cache_key, *args)

        print(f"[JIT] cache miss {cache_key}; tracing and optimizing graph...")
        return self._compile_and_execute_slow_path(
            cache_key,
            use_training_backend,
            disable_conv_bn_fusion,
            *args,
        )

    def train(self):
        was_training = bool(getattr(self.model, "training", False))
        self.model.train()
        if not was_training:
            self.cache.clear()
        return self

    def eval(self):
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        if was_training:
            self.cache.clear()
        return self

    def cuda(self):
        self.model.cuda()
        self.cache.clear()
        return self

    def cpu(self):
        self.model.cpu()
        self.cache.clear()
        return self

    def parameters(self):
        return self.model.parameters()

    def zero_grad(self):
        return self.model.zero_grad()

    def reset_train_profile(self):
        self.last_train_profile = None
        FusedConvBNReLUOp.reset_profile_events()

    def train_backward_profile_summary(self, top_n=10):
        return FusedConvBNReLUOp.profile_summary(top_n=top_n)

    def print_train_backward_profile(self, top_n=10):
        if self.profile:
            print(self.train_backward_profile_summary(top_n=top_n))

    def train_profile_summary(self, top_n=10):
        profile = self.last_train_profile
        if not profile:
            return "[JIT profile] no training profile has been recorded."

        lines = [
            (
                "[JIT profile] "
                f"total={profile['total_ms']:.3f} ms; "
                f"node_run={profile['node_run_ms']:.3f} ms; "
                f"registry_overhead={profile['registry_overhead_ms']:.3f} ms; "
                f"fetch={profile['fetch_ms']:.3f} ms; "
                f"release={profile['release_ms']:.3f} ms"
            ),
            "[JIT profile] by op:",
        ]
        for op_name, item in sorted(
            profile["by_op"].items(),
            key=lambda kv: kv[1]["run_ms"],
            reverse=True,
        ):
            lines.append(
                "  "
                f"{op_name}: calls={item['calls']}, "
                f"run={item['run_ms']:.3f} ms, "
                f"fetch={item['fetch_ms']:.3f} ms, "
                f"release={item['release_ms']:.3f} ms"
            )

        if profile.get("internal_by_op"):
            lines.append("[JIT profile] fused internal forward stages:")
            for op_name, stages in sorted(
                profile["internal_by_op"].items(),
                key=lambda kv: sum(kv[1].values()),
                reverse=True,
            ):
                stage_text = ", ".join(
                    f"{name}={ms:.3f} ms"
                    for name, ms in sorted(stages.items(), key=lambda kv: kv[1], reverse=True)
                )
                lines.append(f"  {op_name}: {stage_text}")

            lines.append(f"[JIT profile] slowest fused internal nodes top {top_n}:")
            for node in sorted(
                profile.get("internal_nodes", []),
                key=lambda item: item["total_ms"],
                reverse=True,
            )[:top_n]:
                stage_text = ", ".join(
                    f"{name}={ms:.3f}"
                    for name, ms in sorted(node["stages"].items(), key=lambda kv: kv[1], reverse=True)
                )
                lines.append(
                    "  "
                    f"{node['op_name']} {node['output_name']}: "
                    f"internal_forward={node['total_ms']:.3f} ms; {stage_text}"
                )

        lines.append(f"[JIT profile] slowest nodes top {top_n}:")
        for node in sorted(profile["nodes"], key=lambda item: item["run_ms"], reverse=True)[:top_n]:
            lines.append(
                "  "
                f"{node['op_name']} {node['output_name']}: "
                f"run={node['run_ms']:.3f} ms, "
                f"fetch={node['fetch_ms']:.3f} ms, "
                f"release={node['release_ms']:.3f} ms, "
                f"inputs={node['input_count']}, releases={node['release_count']}"
            )
        return "\n".join(lines)

    def print_train_profile(self, top_n=10):
        print(self.train_profile_summary(top_n=top_n))

    def _compile_and_execute_slow_path(
        self,
        cache_key,
        use_training_backend,
        disable_conv_bn_fusion,
        *args,
    ):
        snapshot = _snapshot_model_tensors(self.model) if use_training_backend else None

        TracerState._is_tracing = True
        TracerState._current_graph = IRGraph()

        for i, inp in enumerate(args):
            if isinstance(inp, Tensor):
                node = IRNode(f"Input_{i}", [], inp.shape(), str(inp.dtype()))
                TracerState._current_graph.add_node(node)
                inp.ir_node = node

        try:
            model_result = self.model(*args)
            TracerState._current_graph.set_outputs(_collect_output_nodes(model_result))
        finally:
            TracerState._is_tracing = False
            if snapshot is not None:
                _restore_model_tensors(snapshot)

        raw_graph = TracerState._current_graph

        if self.dump_graph:
            print("\n========== Raw Graph ==========")
            print(raw_graph)

        optimized_graph = optimize_graph(
            raw_graph,
            training=use_training_backend,
            disable_stem_fusion=self.disable_stem_fusion,
            disable_conv_bn_fusion=disable_conv_bn_fusion,
        )

        if self.dump_graph:
            print("\n========== Optimized Graph ==========")
            print(optimized_graph)

            raw_count = len(raw_graph.nodes)
            opt_count = len(optimized_graph.nodes)
            fused_count = sum(1 for n in optimized_graph.nodes if n.op_name.startswith("Fused"))
            const_count = sum(1 for n in optimized_graph.nodes if n.op_name == "Constant")

            print("\n========== JIT Optimize Summary ==========")
            print(f"Raw nodes: {raw_count}")
            print(f"Optimized nodes: {opt_count}")
            print(f"Fused nodes: {fused_count}")
            print(f"Constant nodes: {const_count}")

        if use_training_backend:
            train_default_device = "cpu"
            for arg in args:
                if isinstance(arg, Tensor):
                    train_default_device = arg.device()
                    break
            train_plan = self._build_train_plan(optimized_graph, train_default_device)
            self.cache[cache_key] = {
                "backend": "training",
                "executors": [],
                "optimized_graph": optimized_graph,
                "train_plan": train_plan,
            }
            self._print_train_plan_summary(optimized_graph, train_plan)
            return self._execute_train_path(cache_key, *args)

        if not GPU_AVAILABLE:
            raise RuntimeError("JIT CUDA backend requires CuPy, but CuPy is not available.")

        cpp_source = CUDACodegen(optimized_graph).generate()

        node_executors = []
        for node in optimized_graph.nodes:
            if node.op_name.startswith("Input") or node.op_name == "Constant":
                continue

            executor_info = {"node": node}

            if node.op_name in {
                "Conv2dOp",
                "FusedConv2dReLU",
                "FusedConvBNReLU",
                "FusedConvBNAddReLU",
            }:
                kernel_name = node.name.replace("%", "").replace("_", "")
                executor_info["kernel"] = cp.RawKernel(cpp_source, kernel_name)

                out_n, out_c, out_h, out_w = node.output_shape
                tile = CUDACodegen.TILE
                block_dim = (tile, tile, 1)
                grid_dim = (
                    (out_c + tile - 1) // tile,
                    (out_n * out_h * out_w + tile - 1) // tile,
                    1,
                )

                executor_info["grid"] = grid_dim
                executor_info["block"] = block_dim
                executor_info["M"] = out_n * out_h * out_w
                executor_info["out_c"] = out_c

            elif node.op_name == "FusedBNReLU":
                kernel_name = node.name.replace("%", "").replace("_", "")
                executor_info["kernel"] = cp.RawKernel(cpp_source, kernel_name)

                total = _prod(node.output_shape)

                block_dim = (256,)
                grid_dim = ((total + block_dim[0] - 1) // block_dim[0],)

                executor_info["grid"] = grid_dim
                executor_info["block"] = block_dim
                executor_info["total"] = total

            elif node.op_name in {"FusedAddReLU", "Add", "ReLU"}:
                kernel_name = node.name.replace("%", "").replace("_", "")
                executor_info["kernel"] = cp.RawKernel(cpp_source, kernel_name)

                total = _prod(node.output_shape)

                block_dim = (256,)
                grid_dim = ((total + block_dim[0] - 1) // block_dim[0],)

                executor_info["grid"] = grid_dim
                executor_info["block"] = block_dim
                executor_info["total"] = total

            node_executors.append(executor_info)

        self.cache[cache_key] = {
            "backend": "inference",
            "executors": node_executors,
            "optimized_graph": optimized_graph,
        }

        print(f"[JIT] compile success; optimized graph has {len(optimized_graph.nodes)} nodes.")
        return self._execute_fast_path(cache_key, *args)

    @staticmethod
    def _train_tensor_arg(value, tensor_registry, default_device):
        if isinstance(value, IRNode):
            return tensor_registry[value.name]
        if isinstance(value, Tensor):
            return value
        return Tensor(value, device=default_device, requires_grad=False)

    @staticmethod
    def _train_raw_arg(value, tensor_registry):
        if isinstance(value, IRNode):
            return tensor_registry[value.name]
        return value

    @staticmethod
    def _node_attr(node, name, default):
        value = getattr(node, name, None)
        return default if value is None else value

    @staticmethod
    def _train_static_tensor(value, default_device):
        if value is None or isinstance(value, Tensor):
            return value
        return Tensor(value, device=default_device, requires_grad=False)

    @staticmethod
    def _train_fetcher(input_names):
        if not input_names:
            return lambda registry: ()
        if len(input_names) == 1:
            name = input_names[0]
            return lambda registry, name=name: (registry[name],)
        getter = itemgetter(*input_names)
        return lambda registry, getter=getter: getter(registry)

    @staticmethod
    def _train_profile_should_sync(args):
        if cp is None:
            return False
        return any(isinstance(arg, Tensor) and arg.device() == "cuda" for arg in args)

    @staticmethod
    def _train_profile_sync(enabled):
        if enabled:
            cp.cuda.Stream.null.synchronize()

    @staticmethod
    def _profile_op_entry(profile, op_name):
        entry = profile["by_op"].get(op_name)
        if entry is None:
            entry = {
                "calls": 0,
                "fetch_ms": 0.0,
                "run_ms": 0.0,
                "release_ms": 0.0,
            }
            profile["by_op"][op_name] = entry
        return entry

    @staticmethod
    def _train_arg_specs(values):
        input_names = []
        specs = []
        for value in values:
            if isinstance(value, IRNode):
                specs.append(("input", len(input_names)))
                input_names.append(value.name)
            else:
                specs.append(("value", value))
        return tuple(input_names), tuple(specs)

    @staticmethod
    def _train_materialize_args(inputs, specs):
        args = []
        for kind, value in specs:
            if kind == "input":
                args.append(inputs[value])
            else:
                args.append(value)
        return args

    @staticmethod
    def _run_train_fused_conv2d_relu(inputs, params):
        return FusedConv2dReLUOp()(
            inputs[0],
            params["weight"],
            params["bias"],
            params["stride"],
            params["padding"],
            params["dilation"],
            params["groups"],
        )

    @staticmethod
    def _run_train_fused_conv_bn_relu(inputs, params):
        return FusedConvBNReLUOp(
            momentum=params["momentum"],
            eps=params["eps"],
            is_train=params["is_train"],
            profile=params.get("profile", False),
            profile_name=params.get("profile_name"),
        )(
            inputs[0],
            params["weight"],
            params["bias"],
            params["bn_weight"],
            params["bn_bias"],
            params["running_mean"],
            params["running_var"],
            params["stride"],
            params["padding"],
            params["dilation"],
            params["groups"],
        )

    @staticmethod
    def _run_train_fused_conv_bn_add_relu(inputs, params):
        return FusedConvBNAddReLUOp(
            momentum=params["momentum"],
            eps=params["eps"],
            is_train=params["is_train"],
            profile=params.get("profile", False),
            profile_name=params.get("profile_name"),
        )(
            inputs[0],
            inputs[1],
            params["weight"],
            params["bias"],
            params["bn_weight"],
            params["bn_bias"],
            params["running_mean"],
            params["running_var"],
            params["stride"],
            params["padding"],
            params["dilation"],
            params["groups"],
        )

    @staticmethod
    def _run_train_fused_bn_relu(inputs, params):
        return FusedBatchNormReLUOp(
            momentum=params["momentum"],
            eps=params["eps"],
            is_train=params["is_train"],
        )(
            inputs[0],
            params["bn_weight"],
            params["bn_bias"],
            params["running_mean"],
            params["running_var"],
        )

    @staticmethod
    def _run_train_fused_bn_add_relu(inputs, params):
        return FusedBatchNormAddReLUOp(
            momentum=params["momentum"],
            eps=params["eps"],
            is_train=params["is_train"],
        )(
            inputs[0],
            inputs[1],
            params["bn_weight"],
            params["bn_bias"],
            params["running_mean"],
            params["running_var"],
        )

    @staticmethod
    def _run_train_fused_linear_relu(inputs, params):
        return FusedLinearReLUOp()(
            inputs[0],
            params["weight"],
            params["bias"],
        )

    @staticmethod
    def _run_train_fused_add_relu(inputs, params):
        return FusedAddReLUOp()(inputs[0], inputs[1])

    @staticmethod
    def _run_train_fused_mse(inputs, params):
        return FusedMSELossOp()(inputs[0], inputs[1])

    @staticmethod
    def _run_train_fused_cross_entropy(inputs, params):
        return FusedCrossEntropyLossOp()(inputs[0], inputs[1])

    @staticmethod
    def _run_train_conv2d(inputs, params):
        return Conv2dOp()(
            inputs[0],
            params["weight"],
            params["bias"],
            params["stride"],
            params["padding"],
            params["dilation"],
            params["groups"],
        )

    @staticmethod
    def _run_train_batch_norm(inputs, params):
        return BatchNorm2dOp(
            momentum=params["momentum"],
            eps=params["eps"],
            is_train=params["is_train"],
        )(
            inputs[0],
            params["bn_weight"],
            params["bn_bias"],
            params["running_mean"],
            params["running_var"],
        )

    @staticmethod
    def _run_train_matmul_static_rhs(inputs, params):
        return MatMul()(inputs[0], params["rhs"])

    @staticmethod
    def _run_train_add(inputs, params):
        left = inputs[params["left_input"]] if params["left_input"] is not None else params["left"]
        right = inputs[params["right_input"]] if params["right_input"] is not None else params["right"]
        return Add()(left, right)

    @staticmethod
    def _run_train_relu(inputs, params):
        return ReLU()(inputs[0])

    @staticmethod
    def _run_train_reshape(inputs, params):
        return ReshapeOp()(inputs[0], *params["shape"])

    @staticmethod
    def _run_train_sigmoid(inputs, params):
        return Sigmoid()(inputs[0])

    @staticmethod
    def _run_train_elu(inputs, params):
        return ELU(alpha=params["alpha"])(inputs[0])

    @staticmethod
    def _run_train_sum(inputs, params):
        return Sum()(inputs[0])

    @staticmethod
    def _run_train_mse(inputs, params):
        return MSE()(inputs[0], inputs[1])

    @staticmethod
    def _run_train_log_softmax(inputs, params):
        return LogSoftmaxOp()(inputs[0])

    @staticmethod
    def _run_train_nll_loss(inputs, params):
        return NLLLossOp()(inputs[0], inputs[1])

    @staticmethod
    def _run_train_pool(inputs, params):
        return params["op_cls"]()(
            inputs[0],
            params["kernel_size"],
            params["stride"],
            params["padding"],
        )

    @staticmethod
    def _run_train_generic(inputs, params):
        op = params["op_factory"]()
        return op(*CompiledModule._train_materialize_args(inputs, params["arg_specs"]))

    @staticmethod
    def _origin_op_factory(origin_func):
        try:
            origin_func.__class__()
        except TypeError:
            return lambda origin_func=origin_func: origin_func
        op_cls = origin_func.__class__
        return op_cls

    @staticmethod
    def _print_train_plan_summary(graph, train_plan):
        summary = train_plan["summary"]
        fused = summary["fused_counts"]
        print(
            "[JIT] training forward graph executor ready; "
            f"optimized graph has {len(graph.nodes)} nodes; "
            f"executable nodes={summary['executable_nodes']}; "
            f"FusedConvBNReLU={fused.get('FusedConvBNReLU', 0)}; "
            f"FusedConvBNAddReLU={fused.get('FusedConvBNAddReLU', 0)}; "
            f"FusedBNReLU={fused.get('FusedBNReLU', 0)}; "
            f"FusedBNAddReLU={fused.get('FusedBNAddReLU', 0) + fused.get('FusedBatchNormAddReLUOp', 0)}; "
            f"FusedAddReLU={fused.get('FusedAddReLU', 0)}; "
            f"FusedLinearReLU={fused.get('FusedLinearReLU', 0)}; "
            f"fallback_nodes={summary['fallback_nodes']}."
        )
        if summary["fallback_ops"]:
            fallback_ops = ", ".join(
                f"{name}={count}" for name, count in sorted(summary["fallback_ops"].items())
            )
            print(f"[JIT] training fallback ops: {fallback_ops}")

    def _build_train_plan(self, graph, default_device="cpu"):
        output_nodes = graph.outputs or ([graph.nodes[-1]] if graph.nodes else [])
        input_nodes = [node for node in graph.nodes if node.op_name.startswith("Input")]
        constant_nodes = [node for node in graph.nodes if node.op_name == "Constant"]
        input_node_names = tuple(node.name for node in input_nodes)
        output_names = tuple(node.name for node in output_nodes)
        consumer_counts = {
            node.name: count
            for node, count in count_consumers(graph).items()
        }

        train_plan = {
            "input_nodes": input_nodes,
            "input_bindings": tuple((node.name, idx) for idx, node in enumerate(input_nodes)),
            "constant_nodes": constant_nodes,
            "constant_bindings": [],
            "output_nodes": output_nodes,
            "output_names": output_names,
            "consumer_counts": consumer_counts,
            "protected_names": frozenset(input_node_names + output_names),
            "executors": [],
            "steps": [],
            "summary": {
                "executable_nodes": 0,
                "fused_counts": {
                    "FusedConvBNReLU": 0,
                    "FusedConvBNAddReLU": 0,
                    "FusedBNReLU": 0,
                    "FusedBNAddReLU": 0,
                    "FusedBatchNormAddReLUOp": 0,
                    "FusedAddReLU": 0,
                    "FusedLinearReLU": 0,
                },
                "fallback_nodes": 0,
                "fallback_ops": {},
            },
        }

        for node in constant_nodes:
            value = node.constant_data
            if value is not None and not isinstance(value, Tensor):
                value = Tensor(value, device=default_device, requires_grad=False)
            train_plan["constant_bindings"].append((node.name, value))

        summary = train_plan["summary"]

        def is_ir(value):
            return isinstance(value, IRNode)

        def all_static(values):
            return not any(is_ir(value) for value in values)

        def static_tensor(value):
            return self._train_static_tensor(value, default_device)

        def add_step(node, run, input_names, params=None, fallback=False):
            input_names = tuple(input_names)
            params = {} if params is None else params
            step = (
                node.name,
                node.op_name,
                run,
                self._train_fetcher(input_names),
                input_names,
                (),
                params,
            )
            executor = {
                "node": node,
                "out": node.name,
                "input_names": input_names,
                "params": params,
                "run": run,
                "fallback": fallback,
            }
            train_plan["executors"].append(executor)
            train_plan["steps"].append(step)
            summary["executable_nodes"] += 1
            if node.op_name in summary["fused_counts"]:
                summary["fused_counts"][node.op_name] += 1
            if fallback:
                summary["fallback_nodes"] += 1
                summary["fallback_ops"][node.op_name] = summary["fallback_ops"].get(node.op_name, 0) + 1

        def add_generic_step(node, op_factory, values=None, fallback=False):
            values = tuple(node.inputs if values is None else values)
            input_names, arg_specs = self._train_arg_specs(values)
            add_step(
                node,
                self._run_train_generic,
                input_names,
                {
                    "op_factory": op_factory,
                    "arg_specs": arg_specs,
                },
                fallback=fallback,
            )

        for node in graph.nodes:
            if node.op_name.startswith("Input") or node.op_name == "Constant":
                continue

            inputs = tuple(node.inputs)
            op_name = node.op_name

            if op_name in {"FusedConv2dReLU", "FusedConv2dReLUOp"}:
                if is_ir(inputs[0]) and all_static(inputs[1:3]):
                    add_step(
                        node,
                        self._run_train_fused_conv2d_relu,
                        (inputs[0].name,),
                        {
                            "weight": static_tensor(inputs[1]),
                            "bias": static_tensor(inputs[2]),
                            "stride": inputs[3],
                            "padding": inputs[4],
                            "dilation": inputs[5],
                            "groups": inputs[6],
                        },
                    )
                else:
                    add_generic_step(node, FusedConv2dReLUOp)

            elif op_name in {"FusedConvBNReLU", "FusedConvBNReLUOp"}:
                origin = node.origin_func
                momentum = self._node_attr(node, "momentum", getattr(origin, "momentum", 0.1))
                eps = self._node_attr(node, "eps", getattr(origin, "eps", 1e-5))
                is_train = self._node_attr(node, "is_train", getattr(origin, "is_train", True))
                if is_ir(inputs[0]) and all_static(inputs[1:7]):
                    add_step(
                        node,
                        self._run_train_fused_conv_bn_relu,
                        (inputs[0].name,),
                        {
                            "weight": static_tensor(inputs[1]),
                            "bias": static_tensor(inputs[2]),
                            "bn_weight": static_tensor(inputs[3]),
                            "bn_bias": static_tensor(inputs[4]),
                            "running_mean": static_tensor(inputs[5]),
                            "running_var": static_tensor(inputs[6]),
                            "stride": inputs[7],
                            "padding": inputs[8],
                            "dilation": inputs[9],
                            "groups": inputs[10],
                            "momentum": momentum,
                            "eps": eps,
                            "is_train": is_train,
                            "profile": self.profile,
                            "profile_name": node.name,
                        },
                    )
                else:
                    add_generic_step(
                        node,
                        lambda momentum=momentum, eps=eps, is_train=is_train,
                        profile=self.profile, profile_name=node.name: FusedConvBNReLUOp(
                            momentum=momentum,
                            eps=eps,
                            is_train=is_train,
                            profile=profile,
                            profile_name=profile_name,
                        ),
                    )

            elif op_name in {"FusedConvBNAddReLU", "FusedConvBNAddReLUOp"}:
                origin = node.origin_func
                momentum = self._node_attr(node, "momentum", getattr(origin, "momentum", 0.1))
                eps = self._node_attr(node, "eps", getattr(origin, "eps", 1e-5))
                is_train = self._node_attr(node, "is_train", getattr(origin, "is_train", True))
                if is_ir(inputs[0]) and is_ir(inputs[1]) and all_static(inputs[2:8]):
                    add_step(
                        node,
                        self._run_train_fused_conv_bn_add_relu,
                        (inputs[0].name, inputs[1].name),
                        {
                            "weight": static_tensor(inputs[2]),
                            "bias": static_tensor(inputs[3]),
                            "bn_weight": static_tensor(inputs[4]),
                            "bn_bias": static_tensor(inputs[5]),
                            "running_mean": static_tensor(inputs[6]),
                            "running_var": static_tensor(inputs[7]),
                            "stride": inputs[8],
                            "padding": inputs[9],
                            "dilation": inputs[10],
                            "groups": inputs[11],
                            "momentum": momentum,
                            "eps": eps,
                            "is_train": is_train,
                            "profile": self.profile,
                            "profile_name": node.name,
                        },
                    )
                else:
                    add_generic_step(
                        node,
                        lambda momentum=momentum, eps=eps, is_train=is_train,
                        profile=self.profile, profile_name=node.name: FusedConvBNAddReLUOp(
                            momentum=momentum,
                            eps=eps,
                            is_train=is_train,
                            profile=profile,
                            profile_name=profile_name,
                        ),
                    )

            elif op_name in {"FusedBNReLU", "FusedBatchNormReLUOp"}:
                origin = node.origin_func
                momentum = self._node_attr(node, "momentum", getattr(origin, "momentum", 0.1))
                eps = self._node_attr(
                    node,
                    "eps",
                    inputs[5] if len(inputs) > 5 else getattr(origin, "eps", 1e-5),
                )
                is_train = self._node_attr(node, "is_train", getattr(origin, "is_train", True))
                if is_ir(inputs[0]) and all_static(inputs[1:5]):
                    add_step(
                        node,
                        self._run_train_fused_bn_relu,
                        (inputs[0].name,),
                        {
                            "bn_weight": static_tensor(inputs[1]),
                            "bn_bias": static_tensor(inputs[2]),
                            "running_mean": static_tensor(inputs[3]),
                            "running_var": static_tensor(inputs[4]),
                            "momentum": momentum,
                            "eps": eps,
                            "is_train": is_train,
                        },
                    )
                else:
                    add_generic_step(
                        node,
                        lambda momentum=momentum, eps=eps, is_train=is_train: FusedBatchNormReLUOp(
                            momentum=momentum,
                            eps=eps,
                            is_train=is_train,
                        ),
                    )

            elif op_name in {"FusedBNAddReLU", "FusedBatchNormAddReLUOp"}:
                origin = node.origin_func
                momentum = self._node_attr(node, "momentum", getattr(origin, "momentum", 0.1))
                eps = self._node_attr(
                    node,
                    "eps",
                    inputs[6] if len(inputs) > 6 else getattr(origin, "eps", 1e-5),
                )
                is_train = self._node_attr(node, "is_train", getattr(origin, "is_train", True))
                if is_ir(inputs[0]) and is_ir(inputs[1]) and all_static(inputs[2:6]):
                    add_step(
                        node,
                        self._run_train_fused_bn_add_relu,
                        (inputs[0].name, inputs[1].name),
                        {
                            "bn_weight": static_tensor(inputs[2]),
                            "bn_bias": static_tensor(inputs[3]),
                            "running_mean": static_tensor(inputs[4]),
                            "running_var": static_tensor(inputs[5]),
                            "momentum": momentum,
                            "eps": eps,
                            "is_train": is_train,
                        },
                    )
                else:
                    add_generic_step(
                        node,
                        lambda momentum=momentum, eps=eps, is_train=is_train: FusedBatchNormAddReLUOp(
                            momentum=momentum,
                            eps=eps,
                            is_train=is_train,
                        ),
                        values=inputs[:6],
                    )

            elif op_name == "FusedLinearReLU":
                if is_ir(inputs[0]) and all_static(inputs[1:3]):
                    add_step(
                        node,
                        self._run_train_fused_linear_relu,
                        (inputs[0].name,),
                        {
                            "weight": static_tensor(inputs[1]),
                            "bias": static_tensor(inputs[2]),
                        },
                    )
                else:
                    add_generic_step(node, FusedLinearReLUOp)

            elif op_name == "FusedAddReLU":
                if is_ir(inputs[0]) and is_ir(inputs[1]):
                    add_step(
                        node,
                        self._run_train_fused_add_relu,
                        (inputs[0].name, inputs[1].name),
                    )
                else:
                    add_generic_step(node, FusedAddReLUOp)

            elif op_name == "FusedMSELossOp":
                if is_ir(inputs[0]) and is_ir(inputs[1]):
                    add_step(node, self._run_train_fused_mse, (inputs[0].name, inputs[1].name))
                else:
                    add_generic_step(node, FusedMSELossOp)

            elif op_name == "FusedCrossEntropyLossOp":
                if is_ir(inputs[0]) and is_ir(inputs[1]):
                    add_step(node, self._run_train_fused_cross_entropy, (inputs[0].name, inputs[1].name))
                else:
                    add_generic_step(node, FusedCrossEntropyLossOp)

            elif op_name == "Conv2dOp":
                if is_ir(inputs[0]) and all_static(inputs[1:3]):
                    add_step(
                        node,
                        self._run_train_conv2d,
                        (inputs[0].name,),
                        {
                            "weight": static_tensor(inputs[1]),
                            "bias": static_tensor(inputs[2]),
                            "stride": inputs[3],
                            "padding": inputs[4],
                            "dilation": inputs[5],
                            "groups": inputs[6],
                        },
                    )
                else:
                    add_generic_step(node, Conv2dOp)

            elif op_name == "BatchNorm2dOp":
                origin = node.origin_func
                momentum = getattr(origin, "momentum", 0.1)
                eps = getattr(origin, "eps", 1e-5)
                is_train = getattr(origin, "is_train", True)
                if is_ir(inputs[0]) and all_static(inputs[1:5]):
                    add_step(
                        node,
                        self._run_train_batch_norm,
                        (inputs[0].name,),
                        {
                            "bn_weight": static_tensor(inputs[1]),
                            "bn_bias": static_tensor(inputs[2]),
                            "running_mean": static_tensor(inputs[3]),
                            "running_var": static_tensor(inputs[4]),
                            "momentum": momentum,
                            "eps": eps,
                            "is_train": is_train,
                        },
                    )
                else:
                    add_generic_step(
                        node,
                        lambda momentum=momentum, eps=eps, is_train=is_train: BatchNorm2dOp(
                            momentum=momentum,
                            eps=eps,
                            is_train=is_train,
                        ),
                    )

            elif op_name == "MatMul":
                if is_ir(inputs[0]) and all_static(inputs[1:2]):
                    add_step(
                        node,
                        self._run_train_matmul_static_rhs,
                        (inputs[0].name,),
                        {"rhs": static_tensor(inputs[1])},
                    )
                else:
                    add_generic_step(node, MatMul)

            elif op_name == "Add":
                input_names = []
                if is_ir(inputs[0]):
                    left_input = len(input_names)
                    input_names.append(inputs[0].name)
                    left = None
                else:
                    left_input = None
                    left = static_tensor(inputs[0])
                if is_ir(inputs[1]):
                    right_input = len(input_names)
                    input_names.append(inputs[1].name)
                    right = None
                else:
                    right_input = None
                    right = static_tensor(inputs[1])
                add_step(
                    node,
                    self._run_train_add,
                    input_names,
                    {
                        "left_input": left_input,
                        "right_input": right_input,
                        "left": left,
                        "right": right,
                    },
                )

            elif op_name == "ReLU":
                if is_ir(inputs[0]):
                    add_step(node, self._run_train_relu, (inputs[0].name,))
                else:
                    add_generic_step(node, ReLU)

            elif op_name == "ReshapeOp":
                output_shape = node.output_shape
                if is_ir(inputs[0]) and inputs[1:] and all_static(inputs[1:]):
                    shape = tuple(inputs[1:])
                    add_step(node, self._run_train_reshape, (inputs[0].name,), {"shape": shape})
                elif is_ir(inputs[0]) and not inputs[1:]:
                    add_step(node, self._run_train_reshape, (inputs[0].name,), {"shape": tuple(output_shape)})
                else:
                    add_generic_step(node, ReshapeOp)

            elif op_name == "Sigmoid":
                if is_ir(inputs[0]):
                    add_step(node, self._run_train_sigmoid, (inputs[0].name,))
                else:
                    add_generic_step(node, Sigmoid)

            elif op_name == "ELU":
                alpha = getattr(node.origin_func, "alpha", 1.0)
                if is_ir(inputs[0]):
                    add_step(node, self._run_train_elu, (inputs[0].name,), {"alpha": alpha})
                else:
                    add_generic_step(node, lambda alpha=alpha: ELU(alpha=alpha))

            elif op_name == "Sum":
                if is_ir(inputs[0]):
                    add_step(node, self._run_train_sum, (inputs[0].name,))
                else:
                    add_generic_step(node, Sum)

            elif op_name == "MSE":
                if is_ir(inputs[0]) and is_ir(inputs[1]):
                    add_step(node, self._run_train_mse, (inputs[0].name, inputs[1].name))
                else:
                    add_generic_step(node, MSE)

            elif op_name == "LogSoftmaxOp":
                if is_ir(inputs[0]):
                    add_step(node, self._run_train_log_softmax, (inputs[0].name,))
                else:
                    add_generic_step(node, LogSoftmaxOp)

            elif op_name == "NLLLossOp":
                if is_ir(inputs[0]) and is_ir(inputs[1]):
                    add_step(node, self._run_train_nll_loss, (inputs[0].name, inputs[1].name))
                else:
                    add_generic_step(node, NLLLossOp)

            elif op_name in {"MaxPoolOp", "MinPoolOp", "AvgPoolOp"}:
                op_cls = {"MaxPoolOp": MaxPoolOp, "MinPoolOp": MinPoolOp, "AvgPoolOp": AvgPoolOp}[op_name]
                if is_ir(inputs[0]) and all_static(inputs[1:]):
                    add_step(
                        node,
                        self._run_train_pool,
                        (inputs[0].name,),
                        {
                            "op_cls": op_cls,
                            "kernel_size": inputs[1],
                            "stride": inputs[2],
                            "padding": inputs[3] if len(inputs) > 3 else 0,
                        },
                    )
                else:
                    add_generic_step(node, op_cls)

            elif op_name == "ConvTranspose2dOp":
                add_generic_step(node, ConvTranspose2dOp)

            else:
                if node.origin_func is None:
                    raise NotImplementedError(
                        "Training forward graph executor does not support "
                        f"IR op {node.op_name}."
                    )

                add_generic_step(
                    node,
                    self._origin_op_factory(node.origin_func),
                    fallback=True,
                )

        release_remaining = dict(consumer_counts)
        finalized_steps = []
        total_release_slots = 0
        for output_name, op_name, run, fetch_inputs, input_names, _, params in train_plan["steps"]:
            release_names = []
            for input_name in input_names:
                next_count = release_remaining.get(input_name, 0) - 1
                release_remaining[input_name] = next_count
                if next_count <= 0 and input_name not in train_plan["protected_names"]:
                    release_names.append(input_name)
            total_release_slots += len(release_names)
            finalized_steps.append(
                (
                    output_name,
                    op_name,
                    run,
                    fetch_inputs,
                    input_names,
                    tuple(release_names),
                    params,
                )
            )

        train_plan["steps"] = tuple(finalized_steps)
        train_plan["executors"] = tuple(train_plan["executors"])
        summary["release_slots"] = total_release_slots

        return train_plan

    def _execute_train_plan(self, cached_info, *args):
        if self.profile:
            return self._execute_train_plan_profiled(cached_info, *args)

        plan = cached_info["train_plan"]

        tensor_registry = {}
        for name, arg_index in plan["input_bindings"]:
            tensor_registry[name] = args[arg_index]

        for name, value in plan["constant_bindings"]:
            tensor_registry[name] = value

        for output_name, op_name, run, fetch_inputs, input_names, release_names, params in plan["steps"]:
            tensor_registry[output_name] = run(fetch_inputs(tensor_registry), params)
            for input_name in release_names:
                tensor_registry.pop(input_name, None)

        outputs = [tensor_registry[name] for name in plan["output_names"]]
        if len(outputs) == 1:
            result = outputs[0]
        else:
            result = tuple(outputs)
        tensor_registry.clear()
        return result

    def _execute_train_plan_profiled(self, cached_info, *args):
        plan = cached_info["train_plan"]
        sync_cuda = self._train_profile_should_sync(args)
        profile = {
            "total_ms": 0.0,
            "bind_ms": 0.0,
            "fetch_ms": 0.0,
            "node_run_ms": 0.0,
            "release_ms": 0.0,
            "output_ms": 0.0,
            "clear_ms": 0.0,
            "registry_overhead_ms": 0.0,
            "by_op": {},
            "nodes": [],
            "internal_by_op": {},
            "internal_nodes": [],
        }

        FusedConvBNReLUOp.reset_profile_events()
        self._train_profile_sync(sync_cuda)
        total_t0 = time.perf_counter()

        bind_t0 = time.perf_counter()
        tensor_registry = {}
        for name, arg_index in plan["input_bindings"]:
            tensor_registry[name] = args[arg_index]
        for name, value in plan["constant_bindings"]:
            tensor_registry[name] = value
        bind_ms = (time.perf_counter() - bind_t0) * 1000.0
        profile["bind_ms"] = bind_ms

        for output_name, op_name, run, fetch_inputs, input_names, release_names, params in plan["steps"]:
            fetch_t0 = time.perf_counter()
            node_inputs = fetch_inputs(tensor_registry)
            fetch_ms = (time.perf_counter() - fetch_t0) * 1000.0

            self._train_profile_sync(sync_cuda)
            run_t0 = time.perf_counter()
            node_output = run(node_inputs, params)
            tensor_registry[output_name] = node_output
            self._train_profile_sync(sync_cuda)
            run_ms = (time.perf_counter() - run_t0) * 1000.0

            release_t0 = time.perf_counter()
            for input_name in release_names:
                tensor_registry.pop(input_name, None)
            release_ms = (time.perf_counter() - release_t0) * 1000.0

            profile["fetch_ms"] += fetch_ms
            profile["node_run_ms"] += run_ms
            profile["release_ms"] += release_ms

            op_entry = self._profile_op_entry(profile, op_name)
            op_entry["calls"] += 1
            op_entry["fetch_ms"] += fetch_ms
            op_entry["run_ms"] += run_ms
            op_entry["release_ms"] += release_ms

            creator = getattr(node_output, "creator", None)
            profile_stats = getattr(creator, "profile_stats", None)
            forward_stages = profile_stats.get("forward") if profile_stats else None
            if forward_stages:
                stages = dict(forward_stages)
                internal_entry = profile["internal_by_op"].setdefault(op_name, {})
                for stage_name, ms in stages.items():
                    internal_entry[stage_name] = internal_entry.get(stage_name, 0.0) + ms
                profile["internal_nodes"].append(
                    {
                        "output_name": output_name,
                        "op_name": op_name,
                        "total_ms": sum(stages.values()),
                        "stages": stages,
                    }
                )

            profile["nodes"].append(
                {
                    "output_name": output_name,
                    "op_name": op_name,
                    "input_count": len(input_names),
                    "release_count": len(release_names),
                    "fetch_ms": fetch_ms,
                    "run_ms": run_ms,
                    "release_ms": release_ms,
                }
            )

        output_t0 = time.perf_counter()
        outputs = [tensor_registry[name] for name in plan["output_names"]]
        if len(outputs) == 1:
            result = outputs[0]
        else:
            result = tuple(outputs)
        profile["output_ms"] = (time.perf_counter() - output_t0) * 1000.0

        clear_t0 = time.perf_counter()
        tensor_registry.clear()
        profile["clear_ms"] = (time.perf_counter() - clear_t0) * 1000.0

        self._train_profile_sync(sync_cuda)
        profile["total_ms"] = (time.perf_counter() - total_t0) * 1000.0
        profile["registry_overhead_ms"] = (
            profile["bind_ms"]
            + profile["fetch_ms"]
            + profile["release_ms"]
            + profile["output_ms"]
            + profile["clear_ms"]
        )
        self.last_train_profile = profile
        self.print_train_profile()
        return result

    def _execute_train_path(self, cache_key, *args):
        cached_info = self.cache[cache_key]
        if "train_plan" not in cached_info:
            raise RuntimeError("Training cache is missing a pre-bound execution plan.")
        return self._execute_train_plan(cached_info, *args)

    def _execute_fast_path(self, cache_key, *args):
        cached_info = self.cache[cache_key]
        if cached_info.get("backend") == "training":
            return self._execute_train_path(cache_key, *args)

        executors = cached_info["executors"]
        graph = cached_info["optimized_graph"]

        tensor_registry = {}
        buffer_pool = {}
        output_nodes = graph.outputs or ([graph.nodes[-1]] if graph.nodes else [])
        output_names = {node.name for node in output_nodes}

        def alloc(shape, dtype=cp.float32):
            key = (tuple(shape), dtype)
            bucket = buffer_pool.get(key)
            if bucket:
                return bucket.pop()
            return cp.empty(shape, dtype=dtype)

        def release(node):
            if (
                node.name in output_names
                or node.op_name.startswith("Input")
                or node.op_name == "Constant"
                or node.name not in tensor_registry
            ):
                return
            data = tensor_registry.pop(node.name)
            if isinstance(data, cp.ndarray):
                buffer_pool.setdefault((tuple(data.shape), data.dtype), []).append(data)

        input_nodes = [n for n in graph.nodes if n.op_name.startswith("Input")]
        for i, node in enumerate(input_nodes):
            tensor_registry[node.name] = args[i].data

        for node in graph.nodes:
            if node.op_name == "Constant":
                tensor_registry[node.name] = node.constant_data

        remaining = count_consumers(graph)

        def consume_inputs(node):
            for inp in node.inputs:
                if not isinstance(inp, IRNode):
                    continue
                remaining[inp] = remaining.get(inp, 0) - 1
                if remaining[inp] <= 0:
                    release(inp)

        def data_arg(inp):
            if isinstance(inp, IRNode):
                return tensor_registry[inp.name]
            if isinstance(inp, Tensor):
                return inp.data
            return inp

        def tensor_arg(inp):
            if isinstance(inp, IRNode):
                return Tensor(tensor_registry[inp.name], device="cuda")
            return inp

        def flat_param(inp):
            return cp.ascontiguousarray(data_arg(inp).reshape(-1))

        def launch_conv_kernel(node, exec_info, mode):
            if mode == "bn_add_relu":
                x_inp = node.inputs[0]
                identity_inp = node.inputs[1]
                w_inp = node.inputs[2]
                b_inp = node.inputs[3]
                gamma_inp = node.inputs[4]
                beta_inp = node.inputs[5]
                mean_inp = node.inputs[6]
                var_inp = node.inputs[7]
                stride = node.inputs[8]
                padding = node.inputs[9]
                dilation = node.inputs[10]
                groups = node.inputs[11]
            elif mode == "bn_relu":
                x_inp = node.inputs[0]
                w_inp = node.inputs[1]
                b_inp = node.inputs[2]
                gamma_inp = node.inputs[3]
                beta_inp = node.inputs[4]
                mean_inp = node.inputs[5]
                var_inp = node.inputs[6]
                stride = node.inputs[7]
                padding = node.inputs[8]
                dilation = node.inputs[9]
                groups = node.inputs[10]
            else:
                x_inp = node.inputs[0]
                w_inp = node.inputs[1]
                b_inp = node.inputs[2]
                stride = node.inputs[3]
                padding = node.inputs[4]
                dilation = node.inputs[5]
                groups = node.inputs[6]

            x_data = cp.ascontiguousarray(data_arg(x_inp))
            w_data = cp.ascontiguousarray(data_arg(w_inp))

            out_n, out_c, out_h, out_w = node.output_shape
            _, in_c, in_h, in_w = x_data.shape
            _, c_group, kh, kw = w_data.shape
            k_total = c_group * kh * kw

            if b_inp is None:
                b_data = cp.zeros((out_c,), dtype=cp.float32)
            else:
                b_data = flat_param(b_inp)

            stride_h, stride_w = _pair(stride)
            pad_h, pad_w = _pair(padding)
            dil_h, dil_w = _pair(dilation)
            out_data = alloc(node.output_shape, cp.float32)
            eps = float(getattr(node, "eps", 0.0))

            args_prefix = [x_data, w_data, b_data]
            if mode == "bn_add_relu":
                args_prefix.extend([
                    cp.ascontiguousarray(data_arg(identity_inp)),
                    flat_param(gamma_inp),
                    flat_param(beta_inp),
                    flat_param(mean_inp),
                    flat_param(var_inp),
                ])
            elif mode == "bn_relu":
                args_prefix.extend([
                    flat_param(gamma_inp),
                    flat_param(beta_inp),
                    flat_param(mean_inp),
                    flat_param(var_inp),
                ])

            exec_info["kernel"](
                exec_info["grid"],
                exec_info["block"],
                tuple(args_prefix + [
                    out_data,
                    exec_info["M"],
                    out_c,
                    k_total,
                    out_n,
                    in_c,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    kh,
                    kw,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    dil_h,
                    dil_w,
                    groups,
                    eps,
                ]),
            )
            tensor_registry[node.name] = out_data

        for exec_info in executors:
            node = exec_info["node"]

            if node.op_name == "Conv2dOp":
                launch_conv_kernel(node, exec_info, "none")

            elif node.op_name == "FusedConv2dReLU":
                launch_conv_kernel(node, exec_info, "relu")

            elif node.op_name == "FusedConvBNReLU":
                launch_conv_kernel(node, exec_info, "bn_relu")

            elif node.op_name == "FusedConvBNAddReLU":
                launch_conv_kernel(node, exec_info, "bn_add_relu")

            elif node.op_name == "FusedLinearReLU":
                x_data = cp.ascontiguousarray(data_arg(node.inputs[0]))
                w_data = cp.ascontiguousarray(data_arg(node.inputs[1]))
                b_data = data_arg(node.inputs[2])
                out_data = cp.matmul(x_data, w_data)
                out_data = out_data + b_data
                out_data = cp.maximum(out_data, 0)
                tensor_registry[node.name] = cp.ascontiguousarray(out_data)

            elif node.op_name == "FusedBNReLU":
                x_data = cp.ascontiguousarray(tensor_registry[node.inputs[0].name])
                gamma_data = cp.ascontiguousarray(node.inputs[1].data.reshape(-1))
                beta_data = cp.ascontiguousarray(node.inputs[2].data.reshape(-1))
                mean_data = cp.ascontiguousarray(node.inputs[3].data.reshape(-1))
                var_data = cp.ascontiguousarray(node.inputs[4].data.reshape(-1))
                eps = float(node.inputs[5])

                if (
                    x_data.dtype != cp.float32
                    or gamma_data.dtype != cp.float32
                    or beta_data.dtype != cp.float32
                    or mean_data.dtype != cp.float32
                    or var_data.dtype != cp.float32
                ):
                    raise TypeError("FusedBNReLU currently supports only float32 tensors.")

                n, c, h, w = node.output_shape
                out_data = alloc(node.output_shape, cp.float32)

                exec_info["kernel"](
                    exec_info["grid"],
                    exec_info["block"],
                    (
                        x_data,
                        gamma_data,
                        beta_data,
                        mean_data,
                        var_data,
                        out_data,
                        exec_info["total"],
                        n,
                        c,
                        h,
                        w,
                        eps,
                    ),
                )
                tensor_registry[node.name] = out_data

            elif node.op_name == "FusedAddReLU":
                x1_data = cp.ascontiguousarray(tensor_registry[node.inputs[0].name])
                x2_data = cp.ascontiguousarray(tensor_registry[node.inputs[1].name])
                if x1_data.dtype != cp.float32 or x2_data.dtype != cp.float32:
                    raise TypeError("FusedAddReLU currently supports only float32 tensors.")

                out_data = alloc(node.output_shape, cp.float32)
                exec_info["kernel"](
                    exec_info["grid"],
                    exec_info["block"],
                    (x1_data, x2_data, out_data, exec_info["total"]),
                )
                tensor_registry[node.name] = out_data

            elif node.op_name == "MatMul":
                a_data = cp.ascontiguousarray(data_arg(node.inputs[0]))
                b_data = cp.ascontiguousarray(data_arg(node.inputs[1]))
                tensor_registry[node.name] = cp.ascontiguousarray(cp.matmul(a_data, b_data))

            elif node.op_name == "Add":
                a_data = data_arg(node.inputs[0])
                b_data = data_arg(node.inputs[1])
                if (
                    isinstance(node.inputs[0], IRNode)
                    and isinstance(node.inputs[1], IRNode)
                    and tuple(a_data.shape) == tuple(b_data.shape) == tuple(node.output_shape)
                    and a_data.dtype == cp.float32
                    and b_data.dtype == cp.float32
                ):
                    out_data = alloc(node.output_shape, cp.float32)
                    exec_info["kernel"](
                        exec_info["grid"],
                        exec_info["block"],
                        (cp.ascontiguousarray(a_data), cp.ascontiguousarray(b_data), out_data, exec_info["total"]),
                    )
                    tensor_registry[node.name] = out_data
                else:
                    tensor_registry[node.name] = cp.ascontiguousarray(a_data + b_data)

            elif node.op_name == "ReLU":
                x_data = data_arg(node.inputs[0])
                if x_data.dtype == cp.float32 and tuple(x_data.shape) == tuple(node.output_shape):
                    out_data = alloc(node.output_shape, cp.float32)
                    exec_info["kernel"](
                        exec_info["grid"],
                        exec_info["block"],
                        (cp.ascontiguousarray(x_data), out_data, exec_info["total"]),
                    )
                    tensor_registry[node.name] = out_data
                else:
                    tensor_registry[node.name] = cp.ascontiguousarray(cp.maximum(x_data, 0))

            elif node.op_name == "ReshapeOp":
                x_data = data_arg(node.inputs[0])
                tensor_registry[node.name] = cp.ascontiguousarray(x_data.reshape(node.output_shape))

            elif node.op_name == "BatchNorm2dOp":
                x_data = data_arg(node.inputs[0])
                gamma_data = data_arg(node.inputs[1])
                beta_data = data_arg(node.inputs[2])
                running_mean_data = data_arg(node.inputs[3])
                running_var_data = data_arg(node.inputs[4])
                momentum = float(getattr(node.origin_func, "momentum", 0.1))
                eps = float(getattr(node.origin_func, "eps", 1e-5))
                is_train = bool(getattr(node.origin_func, "is_train", True))

                if is_train:
                    mean = cp.mean(x_data, axis=(0, 2, 3), keepdims=True)
                    var = cp.var(x_data, axis=(0, 2, 3), keepdims=True)
                    running_mean_data[...] = momentum * mean + (1.0 - momentum) * running_mean_data
                    running_var_data[...] = momentum * var + (1.0 - momentum) * running_var_data
                else:
                    mean = running_mean_data
                    var = running_var_data

                out_data = gamma_data * (x_data - mean) / cp.sqrt(var + eps) + beta_data
                tensor_registry[node.name] = cp.ascontiguousarray(out_data)

            elif node.op_name == "Sigmoid":
                x_data = data_arg(node.inputs[0])
                tensor_registry[node.name] = cp.ascontiguousarray(1.0 / (1.0 + cp.exp(-x_data)))

            elif node.op_name == "ELU":
                x_data = data_arg(node.inputs[0])
                alpha = float(getattr(node.origin_func, "alpha", 1.0))
                tensor_registry[node.name] = cp.ascontiguousarray(
                    cp.where(x_data > 0, x_data, alpha * (cp.exp(x_data) - 1.0))
                )

            elif node.op_name == "Sum":
                x_data = data_arg(node.inputs[0])
                tensor_registry[node.name] = cp.ascontiguousarray(cp.asarray(cp.sum(x_data)))

            elif node.op_name == "MSE":
                y_pred = data_arg(node.inputs[0])
                y_true = data_arg(node.inputs[1])
                diff = y_pred - y_true
                tensor_registry[node.name] = cp.ascontiguousarray(cp.asarray(cp.mean(diff * diff)))

            elif node.op_name == "LogSoftmaxOp":
                x_data = data_arg(node.inputs[0])
                max_x = cp.max(x_data, axis=1, keepdims=True)
                stable_x = x_data - max_x
                log_sum_exp = cp.log(cp.sum(cp.exp(stable_x), axis=1, keepdims=True))
                tensor_registry[node.name] = cp.ascontiguousarray(stable_x - log_sum_exp)

            elif node.op_name == "NLLLossOp":
                log_probs = data_arg(node.inputs[0])
                target = data_arg(node.inputs[1]).astype(cp.int64).reshape(-1)
                batch = log_probs.shape[0]
                picked = log_probs[cp.arange(batch), target]
                tensor_registry[node.name] = cp.ascontiguousarray(cp.asarray(-cp.mean(picked)))

            else:
                raw_inputs = []
                for inp in node.inputs:
                    if isinstance(inp, IRNode):
                        raw_inputs.append(tensor_registry[inp.name])
                    elif isinstance(inp, Tensor):
                        raw_inputs.append(inp.data)
                    else:
                        raw_inputs.append(inp)

                input_tensors = [
                    Tensor(d, device="cuda") if isinstance(d, cp.ndarray) else d
                    for d in raw_inputs
                ]

                res_tensor = node.origin_func.forward(*input_tensors)
                out_data = res_tensor.data
                if isinstance(out_data, cp.ndarray):
                    out_data = cp.ascontiguousarray(out_data)

                tensor_registry[node.name] = out_data

            consume_inputs(node)

        # Inference-only RawKernel path. Training cache entries return through
        # _execute_train_path above so autograd creators are preserved.
        outputs = [Tensor(tensor_registry[node.name], device="cuda") for node in output_nodes]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


def compile(
    model,
    dump_graph=False,
    training=None,
    profile=False,
    disable_stem_fusion=False,
    disable_conv_bn_fusion=None,
    experimental_conv_bn_fusion=False,
):
    """
    Compile a model wrapper.

    In model.train() mode this is a training forward graph optimizer/executor:
    it preserves autograd creators and relies on eager autograd for backward.
    Training defaults to BN-only fusion (Conv2dOp remains separate); pass
    experimental_conv_bn_fusion=True to enable the larger Conv+BN fusion path.
    In model.eval() mode this uses the inference CUDA RawKernel forward backend.
    """
    return CompiledModule(
        model,
        dump_graph=dump_graph,
        training=training,
        profile=profile,
        disable_stem_fusion=disable_stem_fusion,
        disable_conv_bn_fusion=disable_conv_bn_fusion,
        experimental_conv_bn_fusion=experimental_conv_bn_fusion,
    )


def compile_train(
    model,
    dump_graph=False,
    profile=False,
    disable_stem_fusion=False,
    disable_conv_bn_fusion=None,
    experimental_conv_bn_fusion=False,
):
    """Create a training forward graph optimizer/executor wrapper."""
    return compile(
        model,
        dump_graph=dump_graph,
        training=True,
        profile=profile,
        disable_stem_fusion=disable_stem_fusion,
        disable_conv_bn_fusion=disable_conv_bn_fusion,
        experimental_conv_bn_fusion=experimental_conv_bn_fusion,
    )


def compile_inference(
    model,
    dump_graph=False,
    profile=False,
    disable_stem_fusion=False,
    disable_conv_bn_fusion=None,
    experimental_conv_bn_fusion=False,
):
    """Create an inference-only CUDA RawKernel forward wrapper.

    The caller must put the model in eval mode first. Keeping this explicit
    prevents accidentally running the no-autograd inference backend while
    model.training is still True.
    """
    return compile(
        model,
        dump_graph=dump_graph,
        training=False,
        profile=profile,
        disable_stem_fusion=disable_stem_fusion,
        disable_conv_bn_fusion=disable_conv_bn_fusion,
        experimental_conv_bn_fusion=experimental_conv_bn_fusion,
    )

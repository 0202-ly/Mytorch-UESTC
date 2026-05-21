# mytorch/jit_codegen.py


class CUDACodegen:
    """Generate CUDA kernels for JIT optimized graph nodes.

    Convolution kernels use an implicit-GEMM layout:
      M = N * OH * OW, N = out_channels, K = in_channels_per_group * KH * KW

    Each CUDA block computes a TILE_M x TILE_N output tile and walks K in
    shared-memory tiles. This keeps the convolution computation GEMM-shaped and
    avoids the old one-thread-per-output direct convolution kernel.
    """

    TILE = 16

    def __init__(self, graph):
        self.graph = graph
        self.code_blocks = []

    def generate(self):
        for node in self.graph.nodes:
            if node.op_name.startswith("Input") or node.op_name == "Constant":
                continue

            if node.op_name == "Conv2dOp":
                self.code_blocks.append(self._codegen_conv2d(node, post="none"))
            elif node.op_name == "FusedConv2dReLU":
                self.code_blocks.append(self._codegen_conv2d(node, post="relu"))
            elif node.op_name == "FusedConvBNReLU":
                self.code_blocks.append(self._codegen_conv2d(node, post="bn_relu"))
            elif node.op_name == "FusedConvBNAddReLU":
                self.code_blocks.append(self._codegen_conv2d(node, post="bn_add_relu"))
            elif node.op_name == "Add":
                self.code_blocks.append(self._codegen_add(node))
            elif node.op_name == "ReLU":
                self.code_blocks.append(self._codegen_relu(node))
            elif node.op_name == "FusedBNReLU":
                self.code_blocks.append(self._codegen_fused_bn_relu(node))
            elif node.op_name == "FusedAddReLU":
                self.code_blocks.append(self._codegen_fused_add_relu(node))

        return "\n\n".join(self.code_blocks)

    def _sanitize_kernel_name(self, node):
        return node.name.replace("%", "").replace("_", "")

    def _codegen_add(self, node):
        kernel_name = self._sanitize_kernel_name(node)
        return f"""
extern "C" __global__
void {kernel_name}(
    const float* a,
    const float* b,
    float* output,
    int total
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {{
        output[idx] = a[idx] + b[idx];
    }}
}}
"""

    def _codegen_relu(self, node):
        kernel_name = self._sanitize_kernel_name(node)
        return f"""
extern "C" __global__
void {kernel_name}(
    const float* input,
    float* output,
    int total
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {{
        float x = input[idx];
        output[idx] = x > 0.0f ? x : 0.0f;
    }}
}}
"""

    def _codegen_fused_add_relu(self, node):
        kernel_name = self._sanitize_kernel_name(node)
        return f"""
extern "C" __global__
void {kernel_name}(
    const float* x1,
    const float* x2,
    float* output,
    int total
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {{
        return;
    }}

    float y = x1[idx] + x2[idx];
    output[idx] = y > 0.0f ? y : 0.0f;
}}
"""

    def _codegen_fused_bn_relu(self, node):
        kernel_name = self._sanitize_kernel_name(node)
        return f"""
extern "C" __global__
void {kernel_name}(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int total,
    int N,
    int C,
    int H,
    int W,
    float eps
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {{
        return;
    }}

    int hw = H * W;
    int c = (idx / hw) % C;

    float y = gamma[c] * (input[idx] - running_mean[c]) / sqrtf(running_var[c] + eps) + beta[c];
    output[idx] = y > 0.0f ? y : 0.0f;
}}
"""

    def _codegen_conv2d(self, node, post):
        kernel_name = self._sanitize_kernel_name(node)
        tile = self.TILE

        if post == "none":
            extra_args = ""
            post_body = "float y = acc;"
        elif post == "relu":
            extra_args = ""
            post_body = "float y = acc > 0.0f ? acc : 0.0f;"
        elif post == "bn_relu":
            extra_args = """
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,"""
            post_body = """
    float y = gamma[oc] * (acc - running_mean[oc]) / sqrtf(running_var[oc] + eps) + beta[oc];
    y = y > 0.0f ? y : 0.0f;"""
        elif post == "bn_add_relu":
            extra_args = """
    const float* identity,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,"""
            post_body = """
    int identity_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
    float y = gamma[oc] * (acc - running_mean[oc]) / sqrtf(running_var[oc] + eps) + beta[oc];
    y += identity[identity_idx];
    y = y > 0.0f ? y : 0.0f;"""
        else:
            raise ValueError(f"unknown conv post op: {post}")

        return f"""
extern "C" __global__
void {kernel_name}(
    const float* input,
    const float* weight,
    const float* bias,{extra_args}
    float* output,
    int M,
    int out_c,
    int K,
    int batch,
    int in_c,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dil_h,
    int dil_w,
    int groups,
    float eps
) {{
    __shared__ float As[{tile}][{tile}];
    __shared__ float Bs[{tile}][{tile}];

    int row = blockIdx.y * {tile} + threadIdx.y;
    int oc = blockIdx.x * {tile} + threadIdx.x;

    int out_hw = out_h * out_w;
    int n = row / out_hw;
    int hw = row - n * out_hw;
    int oh = hw / out_w;
    int ow = hw - oh * out_w;

    int out_c_per_group = out_c / groups;
    int in_c_per_group = in_c / groups;
    int group = oc / out_c_per_group;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += {tile}) {{
        int k_a = k0 + threadIdx.x;
        float a = 0.0f;
        if (row < M && k_a < K) {{
            int local_ic = k_a / (kh * kw);
            int rem = k_a - local_ic * kh * kw;
            int ky = rem / kw;
            int kx = rem - ky * kw;
            int ic = group * in_c_per_group + local_ic;
            int iy = oh * stride_h - pad_h + ky * dil_h;
            int ix = ow * stride_w - pad_w + kx * dil_w;
            if (oc < out_c && iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {{
                int input_idx = ((n * in_c + ic) * in_h + iy) * in_w + ix;
                a = input[input_idx];
            }}
        }}

        int k_b = k0 + threadIdx.y;
        float b = 0.0f;
        if (oc < out_c && k_b < K) {{
            int local_ic = k_b / (kh * kw);
            int rem = k_b - local_ic * kh * kw;
            int ky = rem / kw;
            int kx = rem - ky * kw;
            int weight_idx = (((oc * in_c_per_group + local_ic) * kh + ky) * kw + kx);
            b = weight[weight_idx];
        }}

        As[threadIdx.y][threadIdx.x] = a;
        Bs[threadIdx.y][threadIdx.x] = b;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < {tile}; ++kk) {{
            acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
        }}
        __syncthreads();
    }}

    if (row < M && oc < out_c) {{
        if (bias != nullptr) {{
            acc += bias[oc];
        }}
        {post_body}
        int output_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
        output[output_idx] = y;
    }}
}}
"""

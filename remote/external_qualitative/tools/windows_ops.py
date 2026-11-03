"""原生 Windows 推理参考算子；只使用 NumPy 与 PyTorch，不编译外部扩展。"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def voxel_mean(coords, features):
    """保持 HAIS 的首次出现体素顺序、mode=4 均值及事件逆映射。"""
    coords = np.asarray(coords, dtype=np.int64)
    unique, first, inverse, counts = np.unique(
        coords, axis=0, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    inverse = torch.as_tensor(remap[inverse], device=features.device)
    sizes = torch.as_tensor(counts[order], dtype=features.dtype, device=features.device)
    pooled = features.new_zeros((len(order), features.shape[1]))
    # HAIS 先给每个事件乘以 1/count，再累加；不重新归一化输入特征。
    pooled.index_add_(0, inverse, features * sizes.reciprocal()[inverse, None])
    return torch.from_numpy(unique[order]), inverse, pooled


def sparse_batch(path):
    import spconv.pytorch as spconv

    with np.load(path, allow_pickle=False) as data:
        locations = np.column_stack((np.zeros(len(data["ev_loc"]), dtype=np.int64),
                                     data["ev_loc"]))
        features = torch.as_tensor(data["evs_norm"][:, :4], dtype=torch.float32, device="cuda")
    unique, inverse, pooled = voxel_mean(locations, features)
    voxel = spconv.SparseConvTensor(pooled, unique.to(device="cuda", dtype=torch.int32),
                                   [352, 288, 8192], 1)
    return dict(voxel_ev=voxel, p2v_map=inverse, locs=torch.from_numpy(locations))


def segment_csr(src, indptr, out=None, reduce="sum"):
    if out is not None or reduce not in {"mean", "max"}:
        raise NotImplementedError("Only PointEvent inference mean/max without out is supported")
    lengths = indptr[1:] - indptr[:-1]
    if indptr[0].item() != 0 or indptr[-1].item() != len(src) or bool((lengths <= 0).any()):
        raise ValueError("Expected complete, nonempty PointEvent segments")
    return torch.segment_reduce(src, reduce=reduce, lengths=lengths, axis=0)


def _coefficients(value, dim, start, end):
    if value.ndim == 2:
        return value.float()[None, :, :, None]
    if value.ndim == 3:
        return value.float()[:, None, :, start:end]
    if value.ndim == 4 and dim % value.shape[1] == 0:
        return value.float()[:, :, :, start:end].repeat_interleave(dim // value.shape[1], dim=1)
    raise ValueError("Unsupported selective-scan coefficient shape")


@torch.no_grad()
def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    """同一状态递推公式的分块并行前缀扫描；保留跨块状态，不截断上下文。"""
    if A.is_complex() or u.ndim != 3 or not u.shape[-1]:
        raise ValueError("Only nonempty real-valued PointEvent scans are supported")
    dtype = u.dtype
    u, delta, A = u.float(), delta.float(), A.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, length = u.shape
    state = u.new_zeros((batch, dim, A.shape[1]))
    output = []
    for start in range(0, length, 256):
        end = min(start + 256, length)
        steps, inputs = delta[:, :, start:end], u[:, :, start:end]
        transition = torch.exp(steps[:, :, None, :] * A[None, :, :, None])
        value = steps[:, :, None, :] * inputs[:, :, None, :] * _coefficients(B, dim, start, end)
        stride = 1
        # 组合仿射递推 (a,b)：h' = a*h+b；避免逐 token 的大量 GPU 启动。
        while stride < end - start:
            value = torch.cat((value[..., :stride], value[..., stride:]
                               + transition[..., stride:] * value[..., :-stride]), dim=-1)
            transition = torch.cat((transition[..., :stride], transition[..., stride:]
                                    * transition[..., :-stride]), dim=-1)
            stride *= 2
        hidden = value + transition * state[..., None]
        state = hidden[..., -1]
        chunk = (hidden * _coefficients(C, dim, start, end)).sum(dim=2)
        if D is not None:
            chunk = chunk + D.float()[None, :, None] * inputs
        if z is not None:
            chunk = chunk * F.silu(z.float()[:, :, start:end])
        output.append(chunk)
    result = torch.cat(output, dim=-1).to(dtype)
    return (result, state) if return_last_state else result

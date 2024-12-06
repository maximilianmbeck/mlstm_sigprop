import sys
from collections.abc import Callable

import torch
import torch.nn.functional as F

sys.path.append("../..")
from mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v3 import mlstm_chunkwise_max_triton_v3


def mlstm_paper_unstable_fgate(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    mstate_mode: str = "paper",
    mh_layernorm: Callable = None,
) -> torch.Tensor:
    import math

    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    if mstate_mode == "paper":
        vecN = torch.maximum(matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM))  # (B, NH, S, 1)
    elif mstate_mode == "exp_minus_m_to_one":
        vecN = torch.maximum(
            matCtilde.sum(dim=-1, keepdim=True).abs(),
            torch.tensor([1.0], device=_device, dtype=_dtype),
        )  # (B, NH, S, 1)
    elif mstate_mode == "sum_only":
        vecN = matCtilde.sum(dim=-1, keepdim=True).abs()

    elif "denom_one" in mstate_mode:
        split = mstate_mode.split("--")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.tensor([float(denom_const)], device=_device, dtype=_dtype)

    else:
        raise ValueError(f"mstate_mode {mstate_mode} not recognized")

    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    if mh_layernorm is not None:
        with torch.no_grad():
            matH = mh_layernorm(matH.detach())

    return (
        matH,
        vecM.squeeze(-1),
        vecN.squeeze(-1),
        matLogD,
        matLogD_stabilized,
        matD,
        matCtilde,
        matC,
        vecLogSigF,
        vecLogSigF_cumsum,
    )


def mlstm_paper_stable_fgate(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    mstate_mode: str = "paper",
    mh_layernorm: Callable = None,
) -> torch.Tensor:
    import math

    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)

    matLogSigF_tril = vecLogSigF[:, :, :, None].repeat(1, 1, 1, S).tril(-1)
    matLogSigF_cum = matLogSigF_tril.cumsum(-2)

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF_cum, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    if mstate_mode == "paper":
        vecN = torch.maximum(matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM))  # (B, NH, S, 1)
    elif mstate_mode == "exp_minus_m_to_one":
        vecN = torch.maximum(
            matCtilde.sum(dim=-1, keepdim=True).abs(),
            torch.tensor([1.0], device=_device, dtype=_dtype),
        )  # (B, NH, S, 1)
    elif mstate_mode == "sum_only":
        vecN = matCtilde.sum(dim=-1, keepdim=True).abs()

    elif "denom_one" in mstate_mode:
        split = mstate_mode.split("--")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.tensor([float(denom_const)], device=_device, dtype=_dtype)

    else:
        raise ValueError(f"mstate_mode {mstate_mode} not recognized")
    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    if mh_layernorm is not None:
        with torch.no_grad():
            matH = mh_layernorm(matH.detach())

    return (
        matH,
        vecM.squeeze(-1),
        vecN.squeeze(-1),
        matLogD,
        matLogD_stabilized,
        matD,
        matCtilde,
        matC,
        vecLogSigF,
        None,
    )


def mlstm_chunkwise_triton_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    chunk_size: int = 64,
    autocast_kernel_dtype: torch.dtype = torch.float32,
    mh_layernorm: Callable = None,
):
    h_state = mlstm_chunkwise_max_triton_v3(
        q,
        k,
        v,
        i,
        f,
        c_initial,
        n_initial,
        m_initial,
        return_last_states,
        eps,
        chunk_size,
        autocast_kernel_dtype,
    )
    if mh_layernorm is not None:
        with torch.no_grad():
            h_state = mh_layernorm(h_state.detach())
    return h_state


def mlstm_unstable_fgate_ingsig(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    normalization_mode: str = "sum_only",  # "sum_only", "abs_sum", "max_abs_sum_1", "denom_one--1.0"
    mh_layernorm: Callable = None,
):
    import math

    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    # input gates
    vecLogIGate = F.logsigmoid(vecI)

    matLogD = matLogSigF_mask + vecLogIGate[:, :, None, :]

    matD = torch.exp(matLogD)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)

    if normalization_mode == "sum_only":
        vecN = matCtilde.sum(dim=-1, keepdim=True).abs()

    elif normalization_mode == "abs_sum":
        vecN = matCtilde.abs().sum(dim=-1, keepdim=True)

    elif "max_abs_sum_1" in normalization_mode:
        split = normalization_mode.split("--")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.maximum(
            matCtilde.abs().sum(dim=-1, keepdim=True),
            torch.tensor([denom_const], device=_device, dtype=_dtype),
        )

    elif "denom_one" in normalization_mode:
        split = normalization_mode.split("--")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.tensor([float(denom_const)], device=_device, dtype=_dtype)

    else:
        raise ValueError(f"mstate_mode {normalization_mode} not recognized")

    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    if mh_layernorm is not None:
        with torch.no_grad():
            matH = mh_layernorm(matH.detach())

    return (
        matH,
        torch.zeros_like(vecI),
        vecN.squeeze(-1),
        matLogD,
        torch.zeros_like(matLogD),
        matD,
        matCtilde,
        matC,
        vecLogSigF,
        vecLogSigF_cumsum,
    )


def mlstm_stable_fgate_gla(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    tau: float = 1.0,
    eps: float = 1e-6,
    mstate_mode: str = "paper",
    mh_layernorm: Callable = None,
) -> torch.Tensor:
    import math

    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = torch.log(torch.sigmoid(vecF) ** (1 / tau))  # (B, NH, S)

    matLogSigF_tril = vecLogSigF[:, :, :, None].repeat(1, 1, 1, S).tril(-1)
    matLogSigF_cum = matLogSigF_tril.cumsum(-2)

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF_cum, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    if mstate_mode == "paper":
        vecN = torch.maximum(matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM))  # (B, NH, S, 1)
    elif mstate_mode == "exp_minus_m_to_one":
        vecN = torch.maximum(
            matCtilde.sum(dim=-1, keepdim=True).abs(),
            torch.tensor([1.0], device=_device, dtype=_dtype),
        )  # (B, NH, S, 1)
    elif mstate_mode == "sum_only":
        vecN = matCtilde.sum(dim=-1, keepdim=True).abs()

    elif "denom_one" in mstate_mode:
        split = mstate_mode.split("--")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.tensor([float(denom_const)], device=_device, dtype=_dtype)

    else:
        raise ValueError(f"mstate_mode {mstate_mode} not recognized")
    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    if mh_layernorm is not None:
        with torch.no_grad():
            matH = mh_layernorm(matH.detach())

    return (
        matH,
        vecM.squeeze(-1),
        vecN.squeeze(-1),
        matLogD,
        matLogD_stabilized,
        matD,
        matCtilde,
        matC,
        vecLogSigF,
        None,
    )

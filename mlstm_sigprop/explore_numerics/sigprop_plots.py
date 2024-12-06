from functools import partial

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt


def compute_mlstm_outputs(
    mlstm_func,
    B,
    NH,
    S,
    DHQK,
    DHV,
    vecI_offset,
    vecF_offset,
    vecI_init_fn=torch.randn,
    vecF_init_fn=torch.randn,
    q_std=1.0,
    k_std=1.0,
    v_std=1.0,
    DTYPE=torch.float32,
    DEVICE=torch.device("cuda"),
):
    torch.manual_seed(0)

    matQ = q_std * torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE, requires_grad=False)
    matK = k_std * torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE, requires_grad=False)
    matV = v_std * torch.randn((B, NH, S, DHV), dtype=DTYPE, device=DEVICE, requires_grad=False)

    vecI = vecI_offset + vecI_init_fn((B, NH, S), dtype=DTYPE, device=DEVICE, requires_grad=False)
    vecF = vecF_offset + vecF_init_fn((B, NH, S), dtype=DTYPE, device=DEVICE, requires_grad=False)

    out = mlstm_func(matQ, matK, matV, vecI, vecF)

    if isinstance(out, tuple):
        (
            h_out,
            m_out,
            n_out,
            matLogD,
            matLogD_stabilized,
            matD,
            matCtilde,
            matC,
            vecLogSigF,
            vecLogSigF_cumsum,
        ) = out
    else:
        h_out = out
        m_out = None
        n_out = None
        matLogD = None
        matLogD_stabilized = None
        matD = None
        matCtilde = None
        matC = None
        vecLogSigF = None
        vecLogSigF_cumsum = None

    return (
        h_out,
        m_out,
        n_out,
        matLogD,
        matLogD_stabilized,
        matD,
        matCtilde,
        matC,
        vecLogSigF,
        vecLogSigF_cumsum,
    )


def make_offset_sweep(
    mlstm_func,
    B,
    NH,
    S,
    DHQK,
    DHV,
    vecI_offset_range,
    vecF_offset_range,
    q_std=1.0,
    k_std=1.0,
    v_std=1.0,
    vecI_init_fn=torch.randn,
    vecF_init_fn=torch.randn,
    metric: str = "h_out_max_mean",
    DTYPE=torch.float32,
    DEVICE=torch.device("cuda"),
):
    data = []
    data_tensor = torch.zeros(len(vecI_offset_range), len(vecF_offset_range))
    for i, vecI_offset in enumerate(vecI_offset_range):
        for j, vecF_offset in enumerate(vecF_offset_range):
            out = compute_mlstm_outputs(
                mlstm_func,
                B,
                NH,
                S,
                DHQK,
                DHV,
                vecI_offset,
                vecF_offset,
                vecI_init_fn,
                vecF_init_fn,
                q_std,
                k_std,
                v_std,
                DTYPE,
                DEVICE,
            )
            if metric == "h_out_max_mean":
                h_out = out[0]
                h_out_max = h_out.max(-1)[0].mean()
                metric_val = h_out_max

            elif metric == "h_out_max_mean-qk_gain":
                assert q_std == k_std, f"q_std {q_std} != k_std {k_std} for metric {metric}"
                q_input = q_std * torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE, requires_grad=False)
                q_max = q_input.max(-1)[0].mean()
                h_out = out[0]
                h_out_max = h_out.max(-1)[0].mean()
                metric_val = h_out_max / q_max
            elif metric == "h_out_max_mean-v_gain":
                v_input = v_std * torch.randn((B, NH, S, DHV), dtype=DTYPE, device=DEVICE, requires_grad=False)
                v_max = v_input.max(-1)[0].mean()
                h_out = out[0]
                h_out_max = h_out.max(-1)[0].mean()
                metric_val = h_out_max / v_max
            elif metric == "h_out_abs_max_mean-qk_gain":
                assert q_std == k_std, f"q_std {q_std} != k_std {k_std} for metric {metric}"
                q_input = q_std * torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE, requires_grad=False)
                q_max = q_input.abs().max(-1)[0].mean()
                h_out = out[0]
                h_out_max = h_out.abs().max(-1)[0].mean()
                metric_val = h_out_max / q_max
            elif metric == "h_out_abs_max_mean-v_gain":
                v_input = v_std * torch.randn((B, NH, S, DHV), dtype=DTYPE, device=DEVICE, requires_grad=False)
                v_max = v_input.abs().max(-1)[0].mean()
                h_out = out[0]
                h_out_max = h_out.abs().max(-1)[0].mean()
                metric_val = h_out_max / v_max
            else:
                raise ValueError(f"metric {metric} not recognized")
            data_val = {
                "vecI_offset": vecI_offset.item(),
                "vecF_offset": vecF_offset.item(),
                "metric": metric_val.item(),
            }
            data.append(data_val)
            data_tensor[i, j] = metric_val.cpu()

    return data, data_tensor


def make_offset_sweep_meshplot(
    mlstm_func,
    seq_len,
    dqk,
    dv,
    vecI_offset_range,
    vecF_offset_range,
    q_std=1.0,
    k_std=1.0,
    v_std=1.0,
    vecI_init_fn=torch.randn,
    vecF_init_fn=torch.randn,
    levels=np.linspace(0, 10, 10),
    dtype=torch.float32,
    device=torch.device("cuda"),
    metric="h_out_max_mean",
    title_suffix="",
    ax=None,
):
    data, data_tensor = make_offset_sweep(
        mlstm_func=mlstm_func,
        B=1,
        NH=1,
        S=seq_len,
        DHQK=dqk,
        DHV=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        vecI_init_fn=vecI_init_fn,
        vecF_init_fn=vecF_init_fn,
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        metric=metric,
        DTYPE=dtype,
        DEVICE=device,
    )

    if ax is None:
        fig, ax = plt.subplots()

    grid_x, grid_y = torch.meshgrid(vecF_offset_range, vecI_offset_range, indexing="ij")
    grid_x = grid_x.cpu().numpy()
    grid_y = grid_y.cpu().numpy()
    data_z = data_tensor.transpose(0, 1).detach().cpu().numpy()

    # levels = mpl.ticker.MaxNLocator(nbins=20).tick_values(data_z.min(), data_z.max())
    cmap = plt.colormaps["PiYG"]
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = ax.pcolormesh(grid_x, grid_y, data_z, cmap=cmap, norm=norm)
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax)
    ax.set_title(label=f"S={seq_len}, DQK={dqk}, DV={dv}\n{metric} {title_suffix}")
    ax.set_ylabel("vecI_offset")
    ax.set_xlabel("vecF_offset")

    return fig


def make_offset_sweep_meshplot_grid(
    mlstm_func,
    seq_len,
    dqk,
    dv,
    vecI_offset_range,
    vecF_offset_range,
    q_std=1.0,
    k_std=1.0,
    v_std=1.0,
    vecI_init_fn=torch.randn,
    vecF_init_fn=torch.randn,
    levels_before_ln=np.linspace(0, 10, 10),
    levels_after_ln=np.linspace(0, 5, 10),
    dtype=torch.float32,
    norm_eps: list[float] = [1e-5, 1e-6],
    norm_type: str = "layer",
    denom_const_vals: list[float] = [128.0, 2048.0],
    metric="h_out_max_mean",
    device=torch.device("cuda"),
):
    from mlstm_kernels.components.ln import MultiHeadLayerNorm, MultiHeadRMSNorm

    n_ln_eps = len(norm_eps)
    nrows = 3 + len(denom_const_vals)
    fig, axes = plt.subplots(nrows, 1 + n_ln_eps, figsize=(7.5 * (1 + n_ln_eps), nrows * 7.5))

    fig = make_offset_sweep_meshplot(
        mlstm_func=partial(mlstm_func, mstate_mode="paper"),
        seq_len=seq_len,
        dqk=dqk,
        dv=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        dtype=dtype,
        device=device,
        ax=axes[0, 0],
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        vecF_init_fn=vecF_init_fn,
        vecI_init_fn=vecI_init_fn,
        levels=levels_before_ln,
        metric=metric,
        title_suffix="paper before layernorm",
    )
    fig = make_offset_sweep_meshplot(
        mlstm_func=partial(mlstm_func, mstate_mode="exp_minus_m_to_one"),
        seq_len=seq_len,
        dqk=dqk,
        dv=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        dtype=dtype,
        device=device,
        ax=axes[1, 0],
        levels=levels_before_ln,
        metric=metric,
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        vecF_init_fn=vecF_init_fn,
        vecI_init_fn=vecI_init_fn,
        title_suffix="exp_minus_m_to_one before layernorm",
    )
    fig = make_offset_sweep_meshplot(
        mlstm_func=partial(mlstm_func, mstate_mode="denom_one"),
        seq_len=seq_len,
        dqk=dqk,
        dv=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        dtype=dtype,
        device=device,
        ax=axes[2, 0],
        levels=levels_before_ln,
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        vecF_init_fn=vecF_init_fn,
        vecI_init_fn=vecI_init_fn,
        metric=metric,
        title_suffix="denom_one before layernorm",
    )
    for i, denom_const in enumerate(denom_const_vals):
        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(mlstm_func, mstate_mode=f"denom_one--{denom_const}"),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[3 + i, 0],
            levels=levels_before_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"denom_one--{denom_const} before layernorm",
        )
    ## after layernorm
    for i, eps in enumerate(norm_eps):
        if norm_type == "layer":
            mh_norm = MultiHeadLayerNorm(ndim=1 * dv, eps=eps).to(device=device, dtype=dtype)
        elif norm_type == "rms":
            mh_norm = MultiHeadRMSNorm(ndim=1 * dv, eps=eps).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"norm_type {norm_type} not recognized")

        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(mlstm_func, mstate_mode="paper", mh_layernorm=mh_norm),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[0, 1 + i],
            levels=levels_after_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"paper after layernorm\nln_eps={eps}",
        )
        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(
                mlstm_func,
                mstate_mode="exp_minus_m_to_one",
                mh_layernorm=mh_norm,
            ),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[1, 1 + i],
            levels=levels_after_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"exp_minus_m_to_one after layernorm\nln_eps={eps}",
        )
        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(mlstm_func, mstate_mode="denom_one", mh_layernorm=mh_norm),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[2, 1 + i],
            levels=levels_after_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"denom_one after layernorm\nln_eps={eps}",
        )
        for j, denom_const in enumerate(denom_const_vals):
            fig = make_offset_sweep_meshplot(
                mlstm_func=partial(mlstm_func, mstate_mode=f"denom_one--{denom_const}", mh_layernorm=mh_norm),
                seq_len=seq_len,
                dqk=dqk,
                dv=dv,
                vecI_offset_range=vecI_offset_range,
                vecF_offset_range=vecF_offset_range,
                dtype=dtype,
                device=device,
                ax=axes[3 + j, 1 + i],
                levels=levels_after_ln,
                q_std=q_std,
                k_std=k_std,
                v_std=v_std,
                vecF_init_fn=vecF_init_fn,
                vecI_init_fn=vecI_init_fn,
                metric=metric,
                title_suffix=f"denom_one--{denom_const} after layernorm\nln_eps={eps}",
            )

    return fig


# TODO make the plot code more generic (one function for exp and sig input gate)
def make_offset_sweep_meshplot_grid_siging(
    mlstm_func,
    seq_len,
    dqk,
    dv,
    vecI_offset_range,
    vecF_offset_range,
    q_std=1.0,
    k_std=1.0,
    v_std=1.0,
    vecI_init_fn=torch.randn,
    vecF_init_fn=torch.randn,
    levels_before_ln=np.linspace(0, 10, 10),
    levels_after_ln=np.linspace(0, 5, 10),
    dtype=torch.float32,
    norm_eps: list[float] = [1e-5, 1e-6],
    norm_type: str = "layer",
    denom_const_vals: list[float] = [128.0, 2048.0],
    metric="h_out_max_mean",
    device=torch.device("cuda"),
):
    from mlstm_kernels.components.ln import MultiHeadLayerNorm, MultiHeadRMSNorm

    n_ln_eps = len(norm_eps)
    nrows = 3 + len(denom_const_vals)
    fig, axes = plt.subplots(nrows, 1 + n_ln_eps, figsize=(7.5 * (1 + n_ln_eps), nrows * 7.5))

    fig = make_offset_sweep_meshplot(
        mlstm_func=partial(mlstm_func, normalization_mode="max_abs_sum_1"),
        seq_len=seq_len,
        dqk=dqk,
        dv=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        dtype=dtype,
        device=device,
        ax=axes[0, 0],
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        vecF_init_fn=vecF_init_fn,
        vecI_init_fn=vecI_init_fn,
        levels=levels_before_ln,
        metric=metric,
        title_suffix="max_abs_sum_1 before layernorm",
    )
    fig = make_offset_sweep_meshplot(
        mlstm_func=partial(mlstm_func, normalization_mode="abs_sum"),
        seq_len=seq_len,
        dqk=dqk,
        dv=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        dtype=dtype,
        device=device,
        ax=axes[1, 0],
        levels=levels_before_ln,
        metric=metric,
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        vecF_init_fn=vecF_init_fn,
        vecI_init_fn=vecI_init_fn,
        title_suffix="abs_sum before layernorm",
    )
    fig = make_offset_sweep_meshplot(
        mlstm_func=partial(mlstm_func, normalization_mode="sum_only"),
        seq_len=seq_len,
        dqk=dqk,
        dv=dv,
        vecI_offset_range=vecI_offset_range,
        vecF_offset_range=vecF_offset_range,
        dtype=dtype,
        device=device,
        ax=axes[2, 0],
        levels=levels_before_ln,
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
        vecF_init_fn=vecF_init_fn,
        vecI_init_fn=vecI_init_fn,
        metric=metric,
        title_suffix="sum_only before layernorm",
    )
    for i, denom_const in enumerate(denom_const_vals):
        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(mlstm_func, normalization_mode=f"denom_one--{denom_const}"),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[3 + i, 0],
            levels=levels_before_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"denom_one--{denom_const} before layernorm",
        )
    ## after layernorm
    for i, eps in enumerate(norm_eps):
        if norm_type == "layer":
            mh_norm = MultiHeadLayerNorm(ndim=1 * dv, eps=eps).to(device=device, dtype=dtype)
        elif norm_type == "rms":
            mh_norm = MultiHeadRMSNorm(ndim=1 * dv, eps=eps).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"norm_type {norm_type} not recognized")

        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(mlstm_func, normalization_mode="max_abs_sum_1", mh_layernorm=mh_norm),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[0, 1 + i],
            levels=levels_after_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"max_abs_sum_1 after layernorm\nln_eps={eps}",
        )
        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(
                mlstm_func,
                normalization_mode="abs_sum",
                mh_layernorm=mh_norm,
            ),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[1, 1 + i],
            levels=levels_after_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"abs_sum after layernorm\nln_eps={eps}",
        )
        fig = make_offset_sweep_meshplot(
            mlstm_func=partial(mlstm_func, normalization_mode="sum_only", mh_layernorm=mh_norm),
            seq_len=seq_len,
            dqk=dqk,
            dv=dv,
            vecI_offset_range=vecI_offset_range,
            vecF_offset_range=vecF_offset_range,
            dtype=dtype,
            device=device,
            ax=axes[2, 1 + i],
            levels=levels_after_ln,
            q_std=q_std,
            k_std=k_std,
            v_std=v_std,
            vecF_init_fn=vecF_init_fn,
            vecI_init_fn=vecI_init_fn,
            metric=metric,
            title_suffix=f"sum_only after layernorm\nln_eps={eps}",
        )
        for j, denom_const in enumerate(denom_const_vals):
            fig = make_offset_sweep_meshplot(
                mlstm_func=partial(mlstm_func, normalization_mode=f"denom_one--{denom_const}", mh_layernorm=mh_norm),
                seq_len=seq_len,
                dqk=dqk,
                dv=dv,
                vecI_offset_range=vecI_offset_range,
                vecF_offset_range=vecF_offset_range,
                dtype=dtype,
                device=device,
                ax=axes[3 + j, 1 + i],
                levels=levels_after_ln,
                q_std=q_std,
                k_std=k_std,
                v_std=v_std,
                vecF_init_fn=vecF_init_fn,
                vecI_init_fn=vecI_init_fn,
                metric=metric,
                title_suffix=f"denom_one--{denom_const} after layernorm\nln_eps={eps}",
            )

    return fig


def make_h_output_plot_mlstm_with_internals(
    mlstm_func,
    B,
    NH,
    S,
    DHQK,
    DHV,
    vecI_offset,
    vecF_offset,
    seed=0,
    plot_max_min=True,
    vecI_init_fn=torch.randn,
    vecF_init_fn=torch.randn,
    DTYPE=torch.float32,
    DEVICE=torch.device("cuda"),
):
    torch.manual_seed(seed)
    matQ = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
    matK = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
    matV = torch.randn((B, NH, S, DHV), dtype=DTYPE, device=DEVICE)
    # vecI = 0.00001 * torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
    # vecF = -30. + torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
    vecI = vecI_offset + vecI_init_fn((B, NH, S), dtype=DTYPE, device=DEVICE)
    vecF = vecF_offset + vecF_init_fn((B, NH, S), dtype=DTYPE, device=DEVICE)

    out = mlstm_func(matQ, matK, matV, vecI, vecF)

    if isinstance(out, tuple):
        (
            h_out,
            m_out,
            n_out,
            matLogD,
            matLogD_stabilized,
            matD,
            matCtilde,
            matC,
            vecLogSigF,
            vecLogSigF_cumsum,
        ) = out
    else:
        h_out = out
        m_out = None
        n_out = None
        matLogD = None
        matLogD_stabilized = None
        matD = None
        matCtilde = None
        matC = None
        vecLogSigF = None
        vecLogSigF_cumsum = None

    # plot hout + mstate
    h_out_pl_mean = h_out.mean(-1).flatten().cpu().float().numpy()
    h_out_pl_std = h_out.std(-1).flatten().cpu().float().numpy()
    h_out_max = h_out.max(-1)[0].flatten().cpu().float().numpy()
    h_out_min = h_out.min(-1)[0].flatten().cpu().float().numpy()
    if m_out is not None:
        m_pl = m_out.flatten().cpu().float().numpy()
        plt.plot(m_pl, label="m_state")
    # plt.plot(f_pl, label="f_preact")
    # plt.plot(flogsig_pl)
    # plt.plot(n_pl, label="n_state")
    plt.plot(h_out_pl_mean, label="h_out_mean")
    plt.fill_between(
        range(len(h_out_pl_mean)),
        h_out_pl_mean - h_out_pl_std,
        h_out_pl_mean + h_out_pl_std,
        alpha=0.5,
    )
    if plot_max_min:
        plt.plot(h_out_max, label="h_out_max")
        plt.plot(h_out_min, label="h_out_min")

    plt.legend()
    print(f"vecI_offs: {vecI_offset}, vecF_offs: {vecF_offset}")
    print(f"S: {S}, B: {B}, NH: {NH}, DHQK: {DHQK}, DHV: {DHV}")
    # plt.yscale("log")
    plt.show()

    return (
        h_out,
        m_out,
        n_out,
        matLogD,
        matLogD_stabilized,
        matD,
        matCtilde,
        matC,
        vecLogSigF,
        vecLogSigF_cumsum,
    )

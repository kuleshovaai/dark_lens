"""WikiText middle-swap rank that matches the dark swap's NLL damage.

Same chunks, pairing, and residual swap as ``swap_dark_matter_nll.py``:
non-overlapping WikiText chunks in fixed pairs, subspace components exchanged
at matching token positions, then L2 restored. Position 0 is left unchanged.

For each layer window and dark rank ``--null-dim``:

1. Score the unablated model.
2. Swap the trailing ``null_dim`` right-singular directions (the dark band)
   and record the mean NLL and the standard error of that mean across chunks.
3. Probe centered middle bands from rank ``null_dim`` upward. Bands stop
   before the dark tail and then grow to the left. The rank that is kept is
   the smallest probed rank whose mean NLL lies in the dark mean ± ``--tol``.
   Ranks below ``null_dim`` are not scored.
   When ``--tol`` is omitted, the half-width is the dark standard error.
   A window with no such rank is stored as ``middle_unmatched_swap`` and
   the sweep continues. The CSV is rewritten after every window.

The CSV has one row per ablation (``none``, ``dark_swap``, ``middle_swap``).
``band_start`` and ``band_end`` are inclusive singular-vector indices in
descending-``σ`` order.

Example:
  python scripts/wikitext/match_middle_nll.py Qwen3-8B \\
      --layers 20-23 --null-dim 256 --max-chunks 64 --tol 0.05
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "nq_swap"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_dark_matter_generate import layers_tag, parse_layers  # noqa: E402
from src.mech_interp.light_matter import (  # noqa: E402
    swap_subspace_and_renormalize,
    unembedding_right_basis,
)
from src.mech_interp.load_model import load_tl_model  # noqa: E402
from wikitext_utils import build_wikitext_chunks, layer_windows, mode_label  # noqa: E402

load_dotenv()
login(token=os.getenv("HUGGING_FACE_API_KEY"))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mech_interp" / "wikitext_middle_nll_match"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_name", help="Model short name, e.g. Qwen3-8B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="eager")
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Inclusive end (default: n_layers - 2).",
    )
    p.add_argument("--size", type=int, default=4)
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Step between window starts (default: --size).",
    )
    p.add_argument(
        "--layers",
        default=None,
        help="If set, only this layer set (e.g. 20-23); skips the window sweep.",
    )
    p.add_argument(
        "--null-dim",
        type=int,
        nargs="+",
        default=[256],
        help="Dark-band rank(s). The middle rank is searched to match its NLL.",
    )
    p.add_argument("--hook", choices=("resid_post", "resid_mid"), default="resid_post")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-chunks", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tol",
        type=float,
        default=None,
        help=(
            "Half-width of the NLL band around the dark mean. "
            "A middle rank matches when its mean NLL is inside dark NLL ± tol. "
            "Default: the standard error of the dark mean."
        ),
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def middle_band_start(n_sv: int, rank: int, null_dim: int) -> int:
    """Start of a centered band that stops before the trailing dark columns."""
    right_limit = n_sv - null_dim
    start = (n_sv - rank) // 2
    if start + rank <= right_limit:
        return start
    return max(0, right_limit - rank)


def band_bounds(n_sv: int, rank: int, null_dim: int, *, dark: bool) -> tuple[int, int]:
    """Inclusive ``(start, end)`` in the descending singular basis."""
    if dark:
        start = n_sv - null_dim
    else:
        start = middle_band_start(n_sv, rank, null_dim)
    return start, start + rank - 1


def basis_from_vh(
    vh: torch.Tensor, start: int, rank: int, device: torch.device
) -> torch.Tensor:
    """Orthonormal basis ``[d_model, rank]`` from rows ``[start, start+rank)``."""
    return vh[start : start + rank].T.contiguous().to(device=device, dtype=torch.float32)


def make_resid_pair_swap_hook(p_subspace: torch.Tensor) -> Callable:
    """Swap subspace between adjacent pair rows. Position 0 stays put."""

    def hook_fn(resid: torch.Tensor, hook) -> torch.Tensor:
        batch, seq_len, _ = resid.shape
        if batch % 2 != 0:
            raise ValueError(f"Pairwise DM swap needs an even batch size, got {batch}")
        if seq_len <= 1:
            return resid
        perm = (
            torch.arange(batch, device=resid.device).view(-1, 2).flip(1).reshape(-1)
        )
        patched = resid.clone()
        patched[:, 1:, :] = swap_subspace_and_renormalize(
            resid[:, 1:, :], p_subspace, perm
        )
        return patched

    return hook_fn


@torch.no_grad()
def batched_chunk_nll(
    tl_model,
    chunks: torch.Tensor,
    *,
    layers: list[int],
    hook_name: str,
    p_subspace: torch.Tensor | None,
    batch_size: int,
    desc: str,
) -> tuple[float, float]:
    """Mean next-token NLL and accuracy over the packed chunks."""
    n = chunks.shape[0]
    if n % 2 != 0:
        raise ValueError(f"Need an even number of chunks for pairing, got {n}")
    if batch_size < 2 or batch_size % 2 != 0:
        raise ValueError(
            f"--batch-size must be even and >= 2 for pairwise swap, got {batch_size}"
        )

    nll_sum = 0.0
    hit_sum = 0.0
    n_tok = 0
    chunk_nlls: list[torch.Tensor] = []
    device = tl_model.cfg.device
    for start in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        end = min(start + batch_size, n)
        if (end - start) % 2 != 0:
            end -= 1
        if end <= start:
            break
        batch = chunks[start:end].to(device)
        if p_subspace is None:
            logits = tl_model(batch)
        else:
            hook_fn = make_resid_pair_swap_hook(p_subspace)
            fwd_hooks = [(f"blocks.{layer}.{hook_name}", hook_fn) for layer in layers]
            with tl_model.hooks(fwd_hooks=fwd_hooks):
                logits = tl_model(batch)
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = batch[:, 1:]
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        hit = logits[:, :-1, :].argmax(dim=-1) == targets
        nll_sum += float(nll.sum())
        hit_sum += float(hit.sum())
        n_tok += int(nll.numel())
        chunk_nlls.append(nll.mean(dim=-1).detach().cpu())
        del logits, logp, nll, hit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if n_tok == 0:
        raise RuntimeError("No tokens scored")
    chunk_nll = torch.cat(chunk_nlls, dim=0)
    if chunk_nll.shape[0] != n:
        raise RuntimeError(
            f"Scored {chunk_nll.shape[0]} chunks, expected {n}"
        )
    return nll_sum / n_tok, hit_sum / n_tok, chunk_nll


def chunk_mean_se(values: torch.Tensor) -> tuple[float, float]:
    """Mean and standard error of the mean across chunks.

    The standard error is the sample standard deviation divided by ``sqrt(n)``.
    """
    n = int(values.numel())
    if n < 2:
        raise RuntimeError("Need at least 2 chunks to estimate the SE of the mean NLL")
    mean = float(values.mean())
    se = float(values.std(unbiased=True)) / (n ** 0.5)
    return mean, se


def match_middle_rank(
    eval_nll: Callable[[int], float],
    *,
    n_sv: int,
    null_dim: int,
    target_nll: float,
    tol: float,
) -> int | None:
    """Smallest probed middle rank whose mean NLL lies in ``target_nll ± tol``.

    ``eval_nll(rank)`` is the mean NLL of that middle band. The first probe is
    rank ``null_dim``. The search then doubles upward, and binary-searches the
    gap before the first in-band grid point. No rank below ``null_dim`` is
    scored. The returned rank is the smallest scored rank inside the interval,
    or ``None`` when no probed rank falls inside.
    """
    n_avail = n_sv - null_dim
    if n_avail <= 0:
        raise ValueError(f"null_dim={null_dim} leaves no middle directions in {n_sv}")
    if null_dim > n_avail:
        raise ValueError(
            f"null_dim={null_dim} exceeds the {n_avail} directions outside the dark tail"
        )
    if tol < 0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    lo_lim = target_nll - tol
    hi_lim = target_nll + tol
    cache: dict[int, float] = {}

    def nll_of(rank: int) -> float:
        if rank not in cache:
            cache[rank] = eval_nll(rank)
        return cache[rank]

    def status(rank: int) -> int:
        value = nll_of(rank)
        if value < lo_lim - 1e-8:
            return -1
        if value > hi_lim + 1e-8:
            return 1
        return 0

    grid = []
    rank = null_dim
    while rank < n_avail:
        grid.append(rank)
        nxt = rank * 2
        if nxt <= rank:
            break
        rank = nxt
    grid.append(n_avail)

    def refine(left: int, right: int) -> int | None:
        best = right if status(right) == 0 else None
        lo, hi = left, right
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            mid_status = status(mid)
            if mid_status == 0:
                best = mid if best is None else min(best, mid)
                hi = mid
            elif mid_status < 0:
                lo = mid
            else:
                hi = mid
        for edge in (lo, hi):
            if null_dim <= edge <= n_avail and status(edge) == 0:
                best = edge if best is None else min(best, edge)
        return best

    found: int | None = None
    prev = null_dim - 1
    prev_status = -1
    for right in grid:
        right_status = status(right)
        crossed = (prev_status < 0 and right_status > 0) or (
            prev_status > 0 and right_status < 0
        )
        if right_status == 0 or crossed:
            found = refine(prev, right)
            if found is not None:
                break
        prev, prev_status = right, right_status
    return found


def main() -> None:
    args = parse_args()
    null_dims = list(dict.fromkeys(args.null_dim))
    if any(d <= 0 for d in null_dims):
        raise ValueError(f"--null-dim values must be positive, got {args.null_dim}")
    if args.tol is not None and args.tol < 0:
        raise ValueError(f"--tol must be non-negative, got {args.tol}")
    if args.seq_len < 2:
        raise ValueError(f"--seq-len must be >= 2, got {args.seq_len}")
    if args.max_chunks < 2 or args.max_chunks % 2 != 0:
        raise ValueError(
            f"--max-chunks must be even and >= 2 for pairing, got {args.max_chunks}"
        )
    if args.batch_size < 2 or args.batch_size % 2 != 0:
        raise ValueError(
            f"--batch-size must be even and >= 2 for pairing, got {args.batch_size}"
        )

    tl_model = load_tl_model(args.model_name, args.device, args.attn_impl)
    n_layers = tl_model.cfg.n_layers
    if args.layers is not None:
        windows = [parse_layers(args.layers, n_layers)]
    else:
        layer_end = n_layers - 2 if args.layer_end is None else args.layer_end
        stride = args.size if args.stride is None else args.stride
        if args.layer_start < 0 or layer_end >= n_layers:
            raise ValueError(
                f"layer interval [{args.layer_start}, {layer_end}] out of range "
                f"for n_layers={n_layers} (valid indices 0..{n_layers - 1})"
            )
        windows = layer_windows(args.layer_start, layer_end, args.size, stride)
        args.layer_end = layer_end
        args.stride = stride

    hook_name = f"hook_{args.hook}"
    tokenizer = tl_model.tokenizer
    if tokenizer is None:
        raise RuntimeError("Model has no tokenizer; cannot pack WikiText chunks")
    chunks = build_wikitext_chunks(
        tokenizer,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        seq_len=args.seq_len,
        max_chunks=args.max_chunks,
    )
    if chunks.shape[0] % 2 != 0:
        chunks = chunks[:-1]
    if chunks.shape[0] < 2:
        raise ValueError("Need at least 2 WikiText chunks to form a swap pair")

    vh = unembedding_right_basis(tl_model.W_U.detach())
    n_sv = vh.shape[0]
    device = tl_model.W_U.device
    for null_dim in null_dims:
        if null_dim > n_sv:
            raise ValueError(f"--null-dim {null_dim} exceeds n_sv={n_sv}")

    print(
        f"{args.model_name}  {args.dataset}/{args.dataset_config}/{args.split}  "
        f"chunks={chunks.shape[0]} (paired)  seq_len={args.seq_len}  "
        f"windows={len(windows)}  null_dims={null_dims}  n_sv={n_sv}"
    )
    print("Layer windows:", ", ".join(layers_tag(w) for w in windows))

    rows: list[dict] = []

    def score(
        layers: list[int], basis: torch.Tensor | None, desc: str
    ) -> tuple[float, float, torch.Tensor]:
        return batched_chunk_nll(
            tl_model,
            chunks,
            layers=layers,
            hook_name=hook_name,
            p_subspace=basis,
            batch_size=args.batch_size,
            desc=desc,
        )

    _token_nll, none_acc, none_chunks = score([], None, "none")
    none_nll, none_se = chunk_mean_se(none_chunks)
    print(f"  {'none':12s}  NLL={none_nll:.4f}  se={none_se:.4f}  acc={none_acc:.4f}")

    def add_row(
        *,
        ablation: str,
        null_dim: int,
        rank: int,
        band_start: int | None,
        band_end: int | None,
        layers: list[int],
        nll: float,
        acc: float,
        nll_se: float | None,
    ) -> None:
        rows.append(
            {
                "ablation": mode_label(ablation) if ablation != "none" else "none",
                "null_dim": null_dim,
                "subspace_rank": rank,
                "band_start": "" if band_start is None else band_start,
                "band_end": "" if band_end is None else band_end,
                "layers": "" if not layers else layers_tag(layers),
                "layer_start": "" if not layers else layers[0],
                "layer_end": "" if not layers else layers[-1],
                "hook": args.hook,
                "nll": nll,
                "acc": acc,
                "nll_se": "" if nll_se is None else nll_se,
                "delta_nll": nll - none_nll,
                "n_chunks": chunks.shape[0],
                "seq_len": args.seq_len,
                "model": args.model_name,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
            }
        )

    add_row(
        ablation="none",
        null_dim=0,
        rank=0,
        band_start=None,
        band_end=None,
        layers=[],
        nll=none_nll,
        acc=none_acc,
        nll_se=none_se,
    )

    model_tag = args.model_name.replace("/", "_")
    nd_tag = f"nd{null_dims[0]}" if len(null_dims) == 1 else f"nd{null_dims[0]}-{null_dims[-1]}"
    cfg_tag = args.dataset_config.replace("/", "_")
    if args.layers is not None:
        stem = f"{model_tag}_{cfg_tag}_{args.split}_{layers_tag(windows[0])}_{nd_tag}"
    else:
        stem = (
            f"{model_tag}_{cfg_tag}_{args.split}_"
            f"L{args.layer_start}-{args.layer_end}_sz{args.size}_st{args.stride}_{nd_tag}"
        )
    tol_tag = "se_tol" if args.tol is None else f"tol{args.tol:g}"
    out_csv = args.output_dir / f"{stem}_s{args.seq_len}_n{chunks.shape[0]}_nll_match_{tol_tag}.csv"
    unmatched: list[str] = []

    def save_rows() -> None:
        if not rows:
            return
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved {len(rows)} rows to {out_csv}")

    for null_dim in null_dims:
        dark_start, dark_end = band_bounds(n_sv, null_dim, null_dim, dark=True)
        dark_basis = basis_from_vh(vh, dark_start, null_dim, device)
        for layers in windows:
            tag = layers_tag(layers)
            dark_nll, dark_acc, dark_chunks = score(layers, dark_basis, f"{tag} dark")
            dark_mean, dark_se = chunk_mean_se(dark_chunks)
            tol = dark_se if args.tol is None else args.tol
            add_row(
                ablation="dark",
                null_dim=null_dim,
                rank=null_dim,
                band_start=dark_start,
                band_end=dark_end,
                layers=layers,
                nll=dark_mean,
                acc=dark_acc,
                nll_se=dark_se,
            )
            print(
                f"  {'dark_swap':12s}  {tag:12s}  nd={null_dim:<4d}  "
                f"rank={null_dim:<4d}  sv={dark_start}..{dark_end}  "
                f"NLL={dark_mean:.4f}  se={dark_se:.4f}"
            )
            save_rows()

            scored: dict[int, tuple[float, float]] = {}

            def eval_nll(rank: int, _layers=layers, _tag=tag) -> float:
                if rank not in scored:
                    start, end = band_bounds(n_sv, rank, null_dim, dark=False)
                    basis = basis_from_vh(vh, start, rank, device)
                    _nll, acc, mid_chunks = score(_layers, basis, f"{_tag} mid r{rank}")
                    mid_mean = float(mid_chunks.mean())
                    scored[rank] = (mid_mean, acc)
                    print(
                        f"    search  rank={rank:<4d}  sv={start}..{end}  "
                        f"NLL={mid_mean:.4f}"
                    )
                return scored[rank][0]

            mid_rank = match_middle_rank(
                eval_nll,
                n_sv=n_sv,
                null_dim=null_dim,
                target_nll=dark_mean,
                tol=tol,
            )
            if mid_rank is None:
                closest = min(scored, key=lambda r: (abs(scored[r][0] - dark_mean), r))
                mid_nll, mid_acc = scored[closest]
                mid_start, mid_end = band_bounds(n_sv, closest, null_dim, dark=False)
                add_row(
                    ablation="middle_unmatched",
                    null_dim=null_dim,
                    rank=closest,
                    band_start=mid_start,
                    band_end=mid_end,
                    layers=layers,
                    nll=mid_nll,
                    acc=mid_acc,
                    nll_se=None,
                )
                note = (
                    f"{tag} nd={null_dim}: no middle rank in "
                    f"{dark_mean:.4f} ± {tol:.4f}. "
                    f"Closest rank={closest} NLL={mid_nll:.4f}"
                )
                unmatched.append(note)
                print(f"  {'UNMATCHED':12s}  {note}")
                save_rows()
                continue
            mid_start, mid_end = band_bounds(n_sv, mid_rank, null_dim, dark=False)
            mid_nll, mid_acc = scored[mid_rank]
            add_row(
                ablation="middle",
                null_dim=null_dim,
                rank=mid_rank,
                band_start=mid_start,
                band_end=mid_end,
                layers=layers,
                nll=mid_nll,
                acc=mid_acc,
                nll_se=None,
            )
            print(
                f"  {'middle_swap':12s}  {tag:12s}  nd={null_dim:<4d}  "
                f"rank={mid_rank:<4d}  sv={mid_start}..{mid_end}  "
                f"NLL={mid_nll:.4f}  "
                f"dark NLL={dark_mean:.4f} ± {tol:.4f}"
            )
            save_rows()

    save_rows()
    df = pd.DataFrame(rows)
    print(
        df.loc[:, ["ablation", "layers", "null_dim", "subspace_rank", "band_start", "band_end", "nll", "nll_se", "delta_nll"]]
        .to_string(index=False)
    )
    if unmatched:
        band = "dark NLL ± SE" if args.tol is None else f"dark NLL ± {args.tol:g}"
        print(f"{len(unmatched)} window(s) had no middle rank inside the {band} band:")
        for note in unmatched:
            print(f"  {note}")


if __name__ == "__main__":
    main()

"""Average the weights of several fine-tuned rounds ("model soup") into a new checkpoint directory.

All rounds start from the same flan-t5-base weights and share out/tokenizer, so their parameters live in one
basin; a uniform average smooths out the round-to-round swings where each round fixes its target errors and
breaks others.
    uv run python scripts/average_models.py out/konjunktiv-t5.soup-13-16 13 14 15 16
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    out_dir = Path(sys.argv[1])
    rounds = sys.argv[2:]
    # a bare number is a round (out/konjunktiv-t5.round<N>); anything else is a checkpoint directory
    srcs = [Path(f"out/konjunktiv-t5.round{r}") if r.isdigit() else Path(r) for r in rounds]
    total = None
    for src in srcs:
        state = load_file(src / "model.safetensors")
        if total is None:
            total = {k: v.to(torch.float64) for k, v in state.items()}
        else:
            if state.keys() != total.keys():
                raise ValueError(f"{src} has different parameters than {srcs[0]}")
            for k, v in state.items():
                total[k] += v.to(torch.float64)
    avg = {k: (v / len(srcs)).to(torch.float32).contiguous() for k, v in total.items()}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    # config, generation config and tokenizer files are identical across rounds; take them from the last one
    for f in srcs[-1].iterdir():
        if f.is_file() and f.name != "model.safetensors" and f.suffix != ".bin":
            shutil.copy2(f, out_dir / f.name)
    save_file(avg, out_dir / "model.safetensors", metadata={"format": "pt"})
    print(f"wrote {out_dir} = mean of rounds {', '.join(rounds)}")


if __name__ == "__main__":
    main()

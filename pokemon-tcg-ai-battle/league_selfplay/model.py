from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import MemberId
from .features import INPUT_WIDTH
from .numpy_runtime import load_policy, numpy_forward


HIDDEN_ONE = 1024
HIDDEN_TWO = 928
POLICY_PARAMETER_COUNT = 1_478_370


class PolicyValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_one = nn.Linear(INPUT_WIDTH, HIDDEN_ONE)
        self.layer_two = nn.Linear(HIDDEN_ONE, HIDDEN_TWO)
        self.option_head = nn.Linear(HIDDEN_TWO, 1)
        self.stop_head = nn.Linear(HIDDEN_TWO, 1)
        self.value_head = nn.Linear(HIDDEN_TWO, 1)

    def forward(self, features, mask):
        hidden = torch.tanh(self.layer_one(features))
        hidden = torch.tanh(self.layer_two(hidden))
        option_logits = self.option_head(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        stop_logits = self.stop_head(pooled).squeeze(-1)
        values = self.value_head(pooled).squeeze(-1)
        return option_logits, stop_logits, values


def create_population(seed: int, device: str) -> dict[MemberId, PolicyValueNet]:
    torch.manual_seed(seed)
    return {member: PolicyValueNet().to(device) for member in MemberId}


def policy_parameter_count(model: PolicyValueNet) -> int:
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("value_head.")
    )


def export_member(model: PolicyValueNet, path: Path) -> None:
    arrays = {
        "w1": model.layer_one.weight.detach().cpu().numpy().T.astype(np.float32),
        "b1": model.layer_one.bias.detach().cpu().numpy().astype(np.float32),
        "w2": model.layer_two.weight.detach().cpu().numpy().T.astype(np.float32),
        "b2": model.layer_two.bias.detach().cpu().numpy().astype(np.float32),
        "w_option": model.option_head.weight.detach().cpu().numpy()[0].astype(np.float32),
        "b_option": model.option_head.bias.detach().cpu().numpy().astype(np.float32),
        "w_stop": model.stop_head.weight.detach().cpu().numpy()[0].astype(np.float32),
        "b_stop": model.stop_head.bias.detach().cpu().numpy().astype(np.float32),
    }
    np.savez_compressed(path, **arrays)


def parity_smoke(work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "member.npz"
    population = create_population(20260804, "cpu")
    pointers = [next(model.parameters()).data_ptr() for model in population.values()]
    model = population[MemberId.GRIMMSNARL]
    rng = np.random.default_rng(20260804)
    features = rng.normal(0.0, 0.2, size=(7, INPUT_WIDTH)).astype(np.float32)
    try:
        with torch.no_grad():
            tensor = torch.from_numpy(features)[None, :, :]
            mask = torch.ones((1, len(features)), dtype=torch.bool)
            torch_options, torch_stop, _ = model(tensor, mask)
        export_member(model, output)
        weights = load_policy(output)
        numpy_options, numpy_stop = numpy_forward(features, weights)
        option_error = float(
            np.max(np.abs(torch_options[0].numpy() - numpy_options))
        )
        stop_error = abs(float(torch_stop[0]) - numpy_stop)
        return {
            "members": sorted(member.value for member in population),
            "independent_parameter_storage": len(set(pointers)) == len(pointers),
            "policy_parameter_count": policy_parameter_count(model),
            "max_option_logit_error": option_error,
            "max_stop_logit_error": stop_error,
            "all_finite": bool(np.isfinite(option_error) and np.isfinite(stop_error)),
            "mps_available": torch.backends.mps.is_available(),
        }
    finally:
        output.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-smoke", action="store_true")
    args = parser.parse_args()
    if not args.parity_smoke:
        raise SystemExit("only --parity-smoke is supported")
    with tempfile.TemporaryDirectory(prefix="league-model-") as temporary:
        print(json.dumps(parity_smoke(Path(temporary)), sort_keys=True))


if __name__ == "__main__":
    main()

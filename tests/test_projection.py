from __future__ import annotations

import torch

from huav.models.projection import MultiScaleEventProjector, build_view_targets


def _events() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 0.00, 1.0],
            [1.0, 3.0, 0.10, 0.0],
            [1.0, 4.0, 0.10, 1.0],
            [6.0, 4.0, 1.00, 1.0],
        ],
        dtype=torch.float32,
    )


def test_projection_contains_counts_mean_and_latest_time() -> None:
    projector = MultiScaleEventProjector(
        width=8, height=6, time_bins=(4, 2), normalize_counts=False
    )
    views = projector(_events(), torch.zeros(4, dtype=torch.long), batch_size=1)

    assert set(views.xt) == {4, 2}
    assert views.xt[4].shape == (1, 4, 4, 8)
    assert views.yt[4].shape == (1, 4, 4, 6)
    assert views.xt[4][0, 0].sum().item() == 3
    assert views.xt[4][0, 1].sum().item() == 1

    # The t=0 and two t=0.1 events share the cell: mean=0.2/3, latest=0.1.
    assert torch.isclose(views.xt[4][0, 2, 0, 1], torch.tensor(0.2 / 3))
    assert torch.isclose(views.xt[4][0, 3, 0, 1], torch.tensor(0.1))


def test_projection_aggregates_only_the_orthogonal_spatial_axis() -> None:
    projector = MultiScaleEventProjector(
        width=8, height=6, time_bins=(4,), normalize_counts=False
    )
    views = projector(_events(), torch.zeros(4, dtype=torch.long), batch_size=1)

    # Three events share x=1 but have different y, so XT aggregates them.
    assert views.xt[4][0, :2, :, 1].sum().item() == 3
    # The same events remain separated in YT according to y.
    assert views.yt[4][0, :2, 0, 2:5].sum().item() == 3


def test_soft_view_targets_have_local_gaussian_support() -> None:
    labels = torch.tensor([0.0, 0.0, 1.0, 0.0])
    xt, yt = build_view_targets(
        _events(),
        labels,
        torch.zeros(4, dtype=torch.long),
        batch_size=1,
        width=8,
        height=6,
        time_bins=4,
        sigma=1.0,
    )

    assert xt.shape == (1, 1, 4, 8)
    assert yt.shape == (1, 1, 4, 6)
    assert xt.max().item() == 1.0
    assert yt.max().item() == 1.0
    assert 0.0 < xt[0, 0, 0, 0].item() < 1.0

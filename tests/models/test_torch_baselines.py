"""B2 (CNN) and B3 (GRU/LSTM): shapes, causality, masking (plan phase 2 T6)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from adl_etc.data import ppi as P
from adl_etc.models.baselines import CNNBaseline, RNNBaseline, build_model
from adl_etc.models.baselines.cnn import MaskedBatchNorm1d

C = 12


def make_batch(
    b: int = 8, seed: int = 0, lengths: list[int] | None = None
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    ppi_len = torch.tensor(lengths or [3, 30, 7, 1, 12, 30, 5, 9][:b])
    mask = torch.arange(P.K_MAX)[None, :] < ppi_len[:, None]
    cont = torch.randn(b, P.K_MAX, P.PPI_CHANNELS, generator=g) * mask[:, :, None]
    return {"cont": cont, "mask": mask, "ppi_len": ppi_len}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --- shapes and sizes ---------------------------------------------------------------------


def test_cnn_gives_one_prediction_per_flow_and_is_non_causal() -> None:
    m = CNNBaseline(C)

    assert m(make_batch()).shape == (8, C)
    assert m.causal is False and m.views == ("continuous",)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_rnn_gives_a_prediction_after_every_packet_and_is_causal(cell: str) -> None:
    m = RNNBaseline(C, cell=cell)

    assert m(make_batch()).shape == (8, P.K_MAX, C)
    assert m.causal is True


def test_parameter_counts_match_the_spec_sizes() -> None:
    """Spec 005: CNN about 0.4M. GRU/LSTM 'about 0.6M' is low for hidden 256 x 2
    layers (measured 0.73M / 0.96M at 150 classes); the concrete dimensions are kept."""
    assert 0.3e6 < n_params(CNNBaseline(150)) < 0.6e6
    assert 0.5e6 < n_params(RNNBaseline(150, cell="gru")) < 0.9e6
    assert 0.7e6 < n_params(RNNBaseline(150, cell="lstm")) < 1.1e6


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: CNNBaseline(C, channels=(8, 8), kernels=(3,)), "same length"),
        (lambda: CNNBaseline(C, kernels=(4, 5, 3)), "odd"),
        (lambda: RNNBaseline(C, cell="rnn"), "cell"),  # type: ignore[arg-type]
    ],
)
def test_invalid_constructor_arguments_are_rejected(build, message) -> None:
    with pytest.raises(ValueError, match=message):
        build()


# --- masking ----------------------------------------------------------------------------------


def test_cnn_ignores_whatever_sits_in_padded_positions() -> None:
    """Padded positions must not influence the prediction, however they are filled."""
    torch.manual_seed(0)
    m = CNNBaseline(C).eval()
    batch = make_batch()
    tampered = {**batch, "cont": batch["cont"] + 100.0 * (~batch["mask"])[:, :, None]}

    assert torch.allclose(m(batch), m(tampered), atol=1e-5)


def test_cnn_padded_length_does_not_change_a_short_flows_prediction() -> None:
    """A 3-packet flow scores the same alone as inside a batch with 30-packet flows."""
    torch.manual_seed(0)
    m = CNNBaseline(C).eval()
    batch = make_batch()
    alone = {k: v[:1] for k, v in batch.items()}

    assert torch.allclose(m(batch)[0], m(alone)[0], atol=1e-5)


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_rnn_is_causal_a_later_packet_never_changes_an_earlier_output(cell: str) -> None:
    torch.manual_seed(0)
    m = RNNBaseline(C, cell=cell).eval()
    batch = make_batch(lengths=[30] * 8)
    k = 10
    tampered_cont = batch["cont"].clone()
    tampered_cont[:, k:] += torch.randn_like(tampered_cont[:, k:]) * 5  # change packets k+1..30

    a, b = m(batch), m({**batch, "cont": tampered_cont})

    assert torch.allclose(a[:, :k], b[:, :k], atol=1e-6)  # positions 1..k unchanged
    assert not torch.allclose(a[:, k:], b[:, k:])  # and the later ones did move


# --- masked batch norm ----------------------------------------------------------------------------


def test_masked_batch_norm_equals_the_reference_when_nothing_is_masked() -> None:
    torch.manual_seed(0)
    x = torch.randn(6, 5, 11) * 3 + 2
    ones = torch.ones(6, 1, 11)
    ref, ours = nn.BatchNorm1d(5), MaskedBatchNorm1d(5)

    assert torch.allclose(ours(x, ones), ref(x), atol=1e-5)  # training mode
    assert torch.allclose(ours.running_mean, ref.running_mean, atol=1e-6)
    assert torch.allclose(ours.running_var, ref.running_var, atol=1e-5)
    ref.eval(), ours.eval()
    assert torch.allclose(ours(x, ones), ref(x), atol=1e-5)  # eval mode uses the running stats


def test_masked_batch_norm_statistics_come_only_from_real_positions() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 3, 10)
    mask = torch.zeros(4, 1, 10)
    mask[:, :, :4] = 1  # only the first 4 of 10 positions are real
    bn = MaskedBatchNorm1d(3)

    garbage = x.clone()
    garbage[:, :, 4:] = 1e4  # huge junk in the padded positions
    out_clean, out_junk = bn(x, mask), MaskedBatchNorm1d(3)(garbage, mask)

    assert torch.allclose(out_clean[:, :, :4], out_junk[:, :, :4], atol=1e-5)
    real = x[:, :, :4]
    expected = (real - real.mean(dim=(0, 2), keepdim=True)) / torch.sqrt(
        real.var(dim=(0, 2), unbiased=False, keepdim=True) + 1e-5
    )
    assert torch.allclose(out_clean[:, :, :4], expected, atol=1e-4)


def test_an_ordinary_batch_norm_would_be_dominated_by_padding() -> None:
    """Why the masked version exists: on short flows the plain one's statistics are
    mostly padding, so its output on real positions differs materially."""
    torch.manual_seed(0)
    x = torch.randn(8, 3, 30) + 5
    mask = torch.zeros(8, 1, 30)
    mask[:, :, :6] = 1  # ~7 real of 30 slots, like real traffic
    x = x * mask  # padded positions are exactly 0, as after the model's own masking

    masked = MaskedBatchNorm1d(3)(x, mask)[:, :, :6]
    plain = nn.BatchNorm1d(3)(x)[:, :, :6]

    assert (masked.mean() - plain.mean()).abs() > 0.5


# --- the factory ---------------------------------------------------------------------------------


def test_build_model_dispatches_by_name_and_rejects_typos() -> None:
    assert isinstance(build_model({"name": "cnn", "dropout": 0.2}, C), CNNBaseline)
    gru = build_model({"name": "gru", "hidden": 64}, C)
    lstm = build_model({"name": "lstm"}, C)
    assert isinstance(gru, RNNBaseline) and isinstance(lstm.rnn, nn.LSTM)

    with pytest.raises(ValueError, match="unknown model name"):
        build_model({"name": "transformer"}, C)
    with pytest.raises(ValueError, match="unknown model name"):
        build_model({}, C)
    with pytest.raises(ValueError, match="bad arguments"):
        build_model({"name": "cnn", "dropuot": 0.2}, C)  # typo


def test_the_models_train_step_runs_end_to_end_with_the_shared_losses() -> None:
    """A gradient step through each model with its matching loss, so shape or
    masking mistakes surface here rather than inside a long run."""
    from adl_etc.training.losses import cross_entropy_ls, multi_prefix_ce

    batch = make_batch()
    y = torch.randint(0, C, (8,))

    cnn_loss = cross_entropy_ls(CNNBaseline(C)(batch), y)
    gru_loss = multi_prefix_ce(RNNBaseline(C)(batch), y, batch["mask"])

    for loss in (cnn_loss, gru_loss):
        assert torch.isfinite(loss)
        loss.backward()


def test_cnn_output_does_not_depend_on_how_much_padding_a_batch_carries() -> None:
    """Every block re-zeroes padded positions, so the convolutions see the same
    zeros beyond a flow's end whether the batch is padded to 30 or cropped to its
    longest flow. Without that, activations leak into padded positions and a
    batch's padding extent would change its predictions."""
    torch.manual_seed(0)
    m = CNNBaseline(C).eval()
    batch = make_batch()
    longest = int(batch["ppi_len"].max())
    cropped_to = 12  # the longest flow is 30, so use a batch where nothing is that long
    short = {k: v for k, v in batch.items()}
    keep = batch["ppi_len"] <= cropped_to
    short = {k: v[keep] for k, v in short.items()}
    assert longest == 30 and int(short["ppi_len"].max()) <= cropped_to

    padded_to_30 = m(short)
    cropped = m({"cont": short["cont"][:, :cropped_to], "mask": short["mask"][:, :cropped_to]})

    assert torch.allclose(padded_to_30, cropped, atol=1e-5)

"""``evaluation.run`` end to end: config -> checkpoint -> Report + plots (plan T7).

Trains a real (tiny) GRU on synthetic learnable flows so the whole CLI is
exercised for real -- the split loader, `predict_dense`, `EvalArrays`, the
report and the figures -- without needing real D1/D3/D4 data on this machine.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from omegaconf import OmegaConf  # noqa: E402

from adl_etc.data.features import Standardizer  # noqa: E402
from adl_etc.evaluation import artifact as A  # noqa: E402
from adl_etc.evaluation import run as EVAL_RUN  # noqa: E402
from adl_etc.evaluation.report import Report  # noqa: E402
from adl_etc.models.baselines.rnn import RNNBaseline  # noqa: E402
from adl_etc.training.datasets import FlowBatches  # noqa: E402
from adl_etc.training.labels import LabelSpace  # noqa: E402
from adl_etc.training.loop import TrainSettings, fit  # noqa: E402
from tests.training.util import make_flows, make_info, write_shards  # noqa: E402

N_CLASSES = 4


def build_run(tmp_path):
    """Trains a real, tiny GRU on synthetic flows and writes its checkpoint,
    label space and standardizer -- everything an eval config needs."""
    data_root = tmp_path / "data"
    train_data = make_flows(400, N_CLASSES, seed=0)
    val_data = make_flows(100, N_CLASSES, seed=1)
    test_data = make_flows(150, N_CLASSES, seed=2)
    write_shards(data_root, train_data, name="synthetic", period="train", n_classes=N_CLASSES)
    write_shards(data_root, val_data, name="synthetic", period="val", n_classes=N_CLASSES)
    write_shards(data_root, test_data, name="synthetic", period="test", n_classes=N_CLASSES)

    std = Standardizer.fit(write_shards(data_root, train_data, name="std_fit", n_classes=N_CLASSES))
    space = LabelSpace.create(range(N_CLASSES))

    train_fb = FlowBatches(train_data, space, standardizer=std, train=True)
    val_fb = FlowBatches(val_data, space, standardizer=std)
    model = RNNBaseline(N_CLASSES, cell="gru", stem=16, hidden=24, layers=1, dropout=0.0)
    settings = TrainSettings(epochs=25, batch_size=64, lr=5e-3, patience=99, amp=False)
    cfg = OmegaConf.create({"seed": 0, "train": {}})
    info = make_info(tmp_path, cfg)

    run_dir = tmp_path / "run"
    fit(model, train_fb, val_fb, settings=settings, run_dir=run_dir, run_info=info, device="cpu")

    space.save(tmp_path / "label_space.json")
    std.save(tmp_path / "standardizer.json")

    split_yaml = tmp_path / "split.yaml"
    split_yaml.write_text(
        """
dataset: synthetic
temporal: false
splits:
  train: {periods: [train]}
  val: {periods: [val]}
  test: {periods: [test]}
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        f"""
model: {{name: gru, stem: 16, hidden: 24, layers: 1, dropout: 0.0}}
split: {split_yaml.as_posix()}
eval_split: test
data_root: {data_root.as_posix()}
checkpoint: {(run_dir / "best.pt").as_posix()}
label_space: {(tmp_path / "label_space.json").as_posix()}
standardizer: {(tmp_path / "standardizer.json").as_posix()}
out_dir: {out_dir.as_posix()}
seed: 0
device: cpu
eval_cap: 1000
""",
        encoding="utf-8",
    )
    return eval_yaml, out_dir


def test_run_end_to_end_produces_a_report_and_figures(tmp_path):
    eval_yaml, out_dir = build_run(tmp_path)
    rc = EVAL_RUN.main(["--config", str(eval_yaml)])
    assert rc == 0

    eval_dir = out_dir / "eval" / "test"
    for name in (
        A.LOGITS_FILE,
        A.DENSE_LOGITS_FILE,
        A.LABELS_FILE,
        A.PPI_LEN_FILE,
        A.FLOW_INDEX_FILE,
        "report.json",
        "acc_vs_k.png",
        "echo_pareto.png",
        "reliability.png",
    ):
        assert (eval_dir / name).exists(), name

    report = Report.load(eval_dir / "report.json")
    assert report.n_flows == 150
    assert report.n_classes == N_CLASSES
    assert report.split == "test"
    # The task is easy (synthetic, well-separated classes): a real trained
    # GRU should do far better than chance (1/4) by the fully-informed K.
    assert report.acc_at_k[report.ks[-1]] > 0.6
    assert report.echo["pareto_front"]


def test_run_honours_a_cap_smaller_than_the_split(tmp_path):
    eval_yaml, out_dir = build_run(tmp_path)
    text = eval_yaml.read_text().replace("eval_cap: 1000", "eval_cap: 50")
    eval_yaml.write_text(text, encoding="utf-8")

    EVAL_RUN.main(["--config", str(eval_yaml)])
    report = Report.load(out_dir / "eval" / "test" / "report.json")
    ea = A.EvalArrays.load(out_dir / "eval" / "test")
    assert report.n_flows == 50
    assert len(ea.dense) == 50
    assert len(set(ea.flow_index.tolist())) == 50  # a real subset, not padding/repeats


def test_run_writes_logits_that_round_trip_through_eval_arrays(tmp_path):
    eval_yaml, out_dir = build_run(tmp_path)
    EVAL_RUN.main(["--config", str(eval_yaml)])

    ea = A.EvalArrays.load(out_dir / "eval" / "test")
    assert ea.dense.indexing == "effective"  # GRU is causal
    assert len(ea.dense) == 150
    assert set(ea.labels.tolist()) <= set(range(N_CLASSES))


def test_run_report_hash_is_stable_across_two_identical_runs(tmp_path):
    eval_yaml, out_dir = build_run(tmp_path)
    EVAL_RUN.main(["--config", str(eval_yaml)])
    first = json.loads((out_dir / "eval" / "test" / "report.json").read_text())["hash"]

    EVAL_RUN.main(["--config", str(eval_yaml)])  # overwrite, same everything
    second = json.loads((out_dir / "eval" / "test" / "report.json").read_text())["hash"]
    assert first == second


def test_run_refuses_a_checkpoint_from_a_different_label_space(tmp_path):
    eval_yaml, out_dir = build_run(tmp_path)
    # Point at a label space with an extra class: the checkpoint's own hash
    # (recorded at training time) will no longer match.
    bad_space = LabelSpace.create(range(N_CLASSES + 1))
    bad_path = out_dir.parent / "bad_label_space.json"
    bad_space.save(bad_path)
    text = eval_yaml.read_text().replace(
        f'label_space: {(out_dir.parent / "label_space.json").as_posix()}',
        f"label_space: {bad_path.as_posix()}",
    )
    eval_yaml.write_text(text, encoding="utf-8")

    with pytest.raises(Exception, match="label space"):
        EVAL_RUN.main(["--config", str(eval_yaml)])

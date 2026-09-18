import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "scripts/training/train_yolo_candidate.py"
    spec = importlib.util.spec_from_file_location("yolo_training_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_budget_optimizer_accumulation_and_augmentation(runner, tmp_path):
    config = runner.training_arguments(tmp_path / "corpus", tmp_path / "out", "cuda:0")
    assert (config["epochs"], config["batch"], config["imgsz"], config["workers"]) == (5, 2, 576, 2)
    assert config["seed"] == 20260907
    assert config["optimizer"] == "AdamW"
    assert config["lr0"] == 1e-4 and config["lrf"] == 1.0
    assert config["nbs"] / config["batch"] == 4
    assert config["warmup_epochs"] == 0
    assert config["amp"] is False
    assert config["patience"] == 0
    assert all(config[key] == 0 for key in ("mosaic", "mixup", "cutmix", "multi_scale", "scale"))
    assert config["fliplr"] == 0.5 and config["translate"] == 0.1
    assert config["val"] and config["split"] == "val"
    assert config["cls_remap"] is False
    assert config["data"] == str(tmp_path / "corpus/yolo/data.yaml")
    assert config["exist_ok"] is False and config["resume"] is False


def test_native_paths_and_four_class_mapping_verified(runner, tmp_path):
    data = {
        "names": dict(enumerate(runner.CLASSES)),
        "nc": 4,
        "train": str(tmp_path / "yolo/images/train"),
        "val": str(tmp_path / "yolo/images/valid"),
    }
    result = runner.validate_native_data(data, tmp_path)
    assert result["class_names"] == ["player", "goalkeeper", "referee", "ball"]
    assert result["val"].endswith("yolo/images/valid")


@pytest.mark.parametrize(
    "mutation",
    [
        {"val": "/external-450/images"},
        {"train": "/other-train/images"},
        {"test": "/external-450/images"},
        {"download": "download extra labels"},
        {"names": {0: "ball", 1: "player", 2: "referee", 3: "goalkeeper"}},
        {"nc": 80},
    ],
)
def test_native_external_paths_or_class_reordering_rejected(runner, tmp_path, mutation):
    data = {
        "names": dict(enumerate(runner.CLASSES)),
        "nc": 4,
        "train": str(tmp_path / "yolo/images/train"),
        "val": str(tmp_path / "yolo/images/valid"),
        **mutation,
    }
    with pytest.raises(ValueError):
        runner.validate_native_data(data, tmp_path)


def test_exported_image_membership_preserves_partition(runner, tmp_path):
    manifest = {
        "frames": [
            {"image_path": "coco/train/A_1.jpg", "partition": "train"},
            {"image_path": "coco/valid/B_1.jpg", "partition": "valid"},
        ]
    }
    assert runner.expected_image_paths(manifest, tmp_path, "train") == {
        str(tmp_path / "yolo/images/train/A_1.jpg")
    }
    assert runner.expected_image_paths(manifest, tmp_path, "valid") == {
        str(tmp_path / "yolo/images/valid/B_1.jpg")
    }


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), [], [1, float("nan")]])
def test_nonfinite_or_missing_losses_fail(runner, loss):
    with pytest.raises(FloatingPointError):
        runner.finite_loss_values(loss)


@pytest.mark.parametrize(
    "epochs,steps,initial,final",
    [
        ([0, 1, 2, 3], 75, "initial", "final"),
        ([0, 1, 2, 3, 4], 0, "initial", "final"),
        ([0, 1, 2, 3, 4], 93, "same", "same"),
        ([0, 1, 2, 3, 4], 93, None, "final"),
    ],
)
def test_completed_claim_requires_five_epochs_and_verified_update(
    runner, epochs, steps, initial, final
):
    with pytest.raises(RuntimeError):
        runner.training_audit_summary(epochs, steps, initial, final)


def test_completed_training_audit(runner):
    assert runner.training_audit_summary(list(range(5)), 93, "initial", "final") == {
        "epochs_completed": 5,
        "optimizer_steps": 93,
        "classification_head_changed": True,
    }


def test_cli_rejects_unexpected_initialization_and_gpu(runner):
    common = ["--checkpoint", "yolo26m.pt", "--out", "new-run"]
    for extra in (["--device", "cuda:1"], ["--checkpoint-sha256", "0" * 64], ["--epochs", "6"]):
        with pytest.raises(SystemExit):
            runner.parse_args(common + extra)
    assert runner.parse_args(common).checkpoint_sha256 == runner.INITIALIZATION_SHA256

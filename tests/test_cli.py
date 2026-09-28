import json

import pytest
import yaml

from nanovlm.cli.main import main
from nanovlm.config import load_config


def test_preflight_and_no_overwrite(toy_data, tiny_config, tmp_path):
    data, images = toy_data
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(tiny_config.to_dict()))
    report = tmp_path / "preflight.json"
    args = ["preflight", "--config", str(config), "--data-dir", str(data), "--images-root", str(images),
            "--device", "cpu", "--report", str(report), "--decode-all"]
    main(args)
    assert json.loads(report.read_text())["checkpoint_roundtrip"] is True
    with pytest.raises(FileExistsError):
        main(args)


def test_paired_configs_differ_only_in_encoder():
    for kind in ["pilot", "smoke"]:
        a = load_config(f"configs/{kind}_mini_image_conv.yaml").to_dict()
        b = load_config(f"configs/{kind}_mini_patch_conv.yaml").to_dict()
        assert a["model"].pop("encoder") == "image_conv"
        assert b["model"].pop("encoder") == "patch_conv"
        assert a == b

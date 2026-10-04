from septosympto.cli import build_parser, main


def test_list_models_needs_no_images(capsys):
    assert main(["--list-models"]) == 0
    assert "yolo26s-v2" in capsys.readouterr().out


def test_images_are_required_otherwise(capsys):
    try:
        main([])
    except SystemExit as stop:
        assert stop.code == 2
    assert "images" in capsys.readouterr().err


def test_legacy_weights_flag_still_lands_on_necrosis():
    args = build_parser().parse_args(["scans", "-nm", "w.safetensors", "--necrosis-arch", "unet"])
    assert args.necrosis == "w.safetensors"
    assert args.necrosis_arch == "unet"
    assert args.pycnidia == "none"


def test_pycnidia_in_necrosis_is_off_by_default():
    assert build_parser().parse_args(["scans"]).pycnidia_in_necrosis is False


def test_pycnidia_in_necrosis_needs_a_counter(tmp_path, capsys):
    try:
        main([str(tmp_path), "--pycnidia-in-necrosis"])
    except SystemExit as stop:
        assert stop.code == 2
    assert "--pycnidia" in capsys.readouterr().err

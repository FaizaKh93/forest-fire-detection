# Tests de la configuration centralisée.
from src import config


def test_gpu_requirement_can_be_disabled():
    # conftest.py fixe FFD_REQUIRE_GPU=0 : l'import ne doit pas lever d'erreur sans GPU
    assert config.REQUIRE_GPU is False


def test_paths_are_inside_project():
    assert (config.ROOT_DIR / "src" / "config.py").exists()
    for path in [config.DATA_DIR, config.PROCESSED_DIR, config.FEATURES_DIR, config.RESULTS_DIR]:
        assert config.ROOT_DIR in path.parents


def test_constants_are_consistent():
    assert config.TASKS == ["has_smoke", "has_fire"]
    assert config.CLASS_NAMES == {0: "smoke", 1: "fire"}
    assert len(config.IMAGENET_MEAN) == len(config.IMAGENET_STD) == 3
    assert 0 < config.LABELED_FRACTION < 1

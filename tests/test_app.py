# Test de l'application Streamlit (AppTest). Nécessite le Model Registry local (mlflow.db) :
# ignoré en CI, où le registre n'existe pas.
import pytest

from src.config import ROOT_DIR

REGISTRY = ROOT_DIR / "mlflow.db"
pytestmark = pytest.mark.skipif(not REGISTRY.exists(), reason="Model Registry local (mlflow.db) absent")


def test_app_predicts_selected_example():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT_DIR / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not app.exception
    assert len(app.error) == 0 and len(app.success) == 0   # aucun score tant qu'aucun exemple n'est analysé

    app.button(key="AoF07843").click().run()        # exemple « feu et fumée »
    assert not app.exception
    alert_texts = [e.value for e in app.error] + [s.value for s in app.success]
    assert len(alert_texts) == 1                    # scores affichés pour cet exemple seulement
    assert any("Feu et fumée" in t for t in alert_texts)

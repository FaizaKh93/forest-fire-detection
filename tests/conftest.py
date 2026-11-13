# Configuration commune des tests, chargée par pytest avant les fichiers de tests.
import os

# Les tests ne nécessitent pas de GPU : désactive l'obligation définie dans src/config.py
# (doit être fixé avant le premier import de src.config).
os.environ.setdefault("FFD_REQUIRE_GPU", "0")

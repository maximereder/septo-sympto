"""Training harness for SeptoSympto models.

Kept out of the shipped package: a researcher who installs septosympto to run
inference does not need Modal or the training loop. Everything here imports
septosympto, never the other way round. The logic runs locally and is tested
locally; ``modal_app`` is a thin launcher over it, not where the work lives.
"""

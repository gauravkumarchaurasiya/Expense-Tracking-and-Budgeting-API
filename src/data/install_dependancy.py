# install_dependencies.py
import subprocess
import sys

def download_spacy_model():
    """Download the en_core_web_sm spaCy model."""
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
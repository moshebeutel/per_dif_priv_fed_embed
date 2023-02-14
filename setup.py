"""Package configuration."""
from setuptools import find_packages, setup

setup(
    name="per_dif_priv_fed_embed",
    version="0.0.1",
    packages=find_packages(where="per_dif_priv_fed_embed"),
    package_dir={"": "per_dif_priv_fed_embed"},
)
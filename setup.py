"""
this setup will check for dependencies and install TrajPy on your computer
"""
from pathlib import Path

import tomllib
from setuptools import find_packages, setup

# Read pyproject.toml
pyproject_path = Path(__file__).parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

# Extract project metadata
project = pyproject_data["project"]
poetry_deps = pyproject_data["tool"]["poetry"]["dependencies"]

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Convert poetry dependencies to setuptools format
install_requires = [
    f"{name} {version}" if not isinstance(version, dict) else name
    for name, version in poetry_deps.items()
    if name != "python"
]

setup(
    name=project["name"],
    version=project["version"],
    url=project["urls"]["homepage"],
    author=project["authors"][0]["name"],
    author_email=project["authors"][0]["email"],
    description=project["description"],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=project.get("keywords", []),
    license=project["license"],
    python_requires=project["requires-python"],
    packages=find_packages(),
    install_requires=install_requires,
)


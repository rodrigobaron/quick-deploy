from setuptools import find_namespace_packages, setup, find_packages
import re


VERSION = open('VERSION').read()
LONG_DESCRIPTION = open('README.md').read()

_deps = [
    "numpy==1.21.5",
    "pyaml==21.10.1",
    "types-PyYAML==6.0.1",
    "onnxruntime==1.10.0",
    "onnxruntime-tools==1.7.0",
    "onnxmltools==1.10.0",
    "sympy==1.9.0",
    "unidecode==1.3.2",
    "torch==1.10.1",
    "transformers==4.15.0",
    "scikit-learn==1.0.1",
    "skl2onnx==1.9.0",
    "tensorflow>=2.3",
    "tf2onnx==1.9.3",
    "xgboost==1.5.1",
    "pytest==6.2.5",
    "pytest-cov==3.0.0",
    "mock==4.0.3",
    "black==21.12b0",
    "flake8==4.0.1",
    "mypy==0.930",
    "isort==5.10.1",
    "coverage-badge==1.1.0",
    "python-build==0.2.13",
    "build==0.7.0"
]


deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

extras = {}

extras["torch"] = deps_list("torch")
extras["tf"] = deps_list("tensorflow", "tf2onnx")
extras["transformers"] = (
    deps_list("transformers")
    + extras["torch"]
    + extras["tf"] 
)
extras["sklearn"] = deps_list("scikit-learn", "skl2onnx")
extras["xgboost"] = (
    deps_list("xgboost")
    + extras["sklearn"] 
)

extras["all"] = (
    extras["torch"]
    + extras["tf"]
    + extras["transformers"]
    + extras["sklearn"]
    + extras["xgboost"]
)

extras["dev"] = (
    deps_list(
        "pytest",
        "pytest-cov",
        "mock",
        "black",
        "flake8",
        "mypy",
        "isort"
    )
    + extras["all"] 
)

extras["build"] = (
    deps_list(
        "python-build",
        "coverage-badge",
        "build"
    )
    + extras["dev"]
)

install_requires = [
    deps["numpy"],
    deps["pyaml"],
    deps["types-PyYAML"],
    deps["onnxruntime"],
    deps["onnxruntime-tools"],
    deps["onnxmltools"],
    deps["sympy"],
    deps["unidecode"]
]


setup(
    name='quick-deploy',
    version=VERSION,
    description='Quick-Deploy optimize and deploy Machine Learning models as fast inference API.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/rodrigobaron/quick-deploy",
    author="Rodrigo Baron",
    author_email="baron.rodrigo0@gmail.com",
    license="Apache",
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    package_data={
        'quick_deploy': ['py.typed'],
    },
    zip_safe=False,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'quick-deploy = quick_deploy.cli:main',
        ],
    },
    data_files=[('', ['VERSION', 'README.md', 'LICENSE'])],
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

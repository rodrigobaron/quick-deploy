from setuptools import find_namespace_packages, setup

VERSION = open('VERSION').read()
LONG_DESCRIPTION = open('README.md').read()

setup(
    name='fast-deploy',
    version=VERSION,
    description='FastDeploy optimize and deploy your Machine Learning models as API.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/rodrigobaron/fastdeploy",
    author="Rodrigo Baron",
    author_email="baron.rodrigo0@gmail.com",
    license="MIT",
    packages=find_namespace_packages(include=['fast_deploy.*'], where="src"),
    package_dir={'': 'src'},
    package_data={
        'fast_deploy': ['py.typed'],
    },
    data_files=[('', ['VERSION', 'README.md', 'LICENSE'])],
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=[
    ],
    zip_safe=False
)

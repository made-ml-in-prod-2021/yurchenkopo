from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Heart disease prediction project.',
    author='YurchenkoPO',
	install_requires=[
		'omegaconf==2.1.0.dev26',
		'hydra==2.5',
		'hydra-core==1.1.0.dev6',
        'python-dotenv>=0.5.1',
        'scikit-learn==0.24.1',
        'dataclasses==0.6',
        'pyyaml==5.4.1',
        'marshmallow-dataclass==8.4.1',
        'pandas==1.2.4',
		'numpy==1.20.2',
		'pytest==6.2.4',
    ],
    license='MIT',
)

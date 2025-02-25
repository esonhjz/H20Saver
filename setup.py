from setuptools import setup, find_packages

setup(
    name='H20Saver',
    version='1.0.0',
    author='Eason Huang',
    author_email='easonhuangjz@outlook.com',
    description='A YOLO-based drowning detection project',
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.204',
        'torch>=2.0',
        'opencv-python>=4.5',
        'numpy>=1.24.4',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.1',
        'seaborn>=0.12.2',
        'tqdm>=4.66.1',
        'PyYAML>=6.0',
        'protobuf>=3.19.6',
        'pillow>=9.4.0',
        'click>=8.1.6',
        'torchvision>=0.15.2',
        'scipy>=1.10.1',
        'pandas>=2.0.3'
    ],
    include_package_data=True,
)
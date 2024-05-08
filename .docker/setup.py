from setuptools import setup


setup(
    name='yolo-ouster',
    version='0.1.0',
    packages=[
        'models',
        'models.hub',
        'utils',
        'utils.aws',
        'utils.flask_rest_api',
        'utils.google_app_engine',
        'utils.loggers'
    ],
    install_requires=[
        'matplotlib',
        'numpy',
        'opencv-python',
        'Pillow',
        'PyYAML',
        'requests',
        'scipy',
        'tqdm',
        'pandas',
        'seaborn',
        'thop'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)

from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='xarray-simlab-ode',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Xarray-simlab extension for building and running models based on ordinary differential equations",
    license="BSD",
    author="Benjamin Post",
    author_email='ben@anoutpost.com',
    url='https://github.com/ben1post/xarray-simlab-ode',
    packages=['xso'],
    entry_points={
        'console_scripts': [
            'xso=xso.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='xarray-simlab-ode',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)

from setuptools import setup

package_name = 'llm_velocity_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leeyunjae66',
    maintainer_email='210678775+leeyunjae66@users.noreply.github.com',
    description='Natural-language velocity control via a local LLM (Ollama).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'velocity_node = llm_velocity_controller.velocity_node:main',
        ],
    },
)

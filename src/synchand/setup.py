from setuptools import find_packages, setup

package_name = 'synchand'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='clark',
    maintainer_email='jeonghuibag384@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'talker = synchand.angle_publisher:main',
            'listener = synchand.angle_subscriber:main',
            'vision = synchand.vision_publisher:main',
            'vision_15dof = synchand.vision_publisher_15dof:main', # 👇 추가된 15자유도 노드!
            'vision_world_15dof = synchand.vision_world_15dof:main',
        ],
    },
)

import os
import platform
import socket


def detect_environment():
    """Detect if running on server or local PC"""

    env_info = {
        'is_server': False,
        'is_local': False,
        'is_colab': False,
        'is_kaggle': False,
        'hostname': socket.gethostname(),
        'platform': platform.system(),
        'cpu_count': os.cpu_count(),
    }

    # Check for cloud/serving environments
    if 'COLAB_GPU' in os.environ:
        env_info['is_colab'] = True
        env_info['is_server'] = True
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        env_info['is_kaggle'] = True
        env_info['is_server'] = True

    # Check for server indicators
    server_indicators = [
        'SERVER', 'PRODUCTION', 'CLOUD', 'AWS', 'GCP',
        'AZURE', 'KUBERNETES', 'DOCKER', 'SLURM'
    ]

    for indicator in server_indicators:
        if indicator in os.environ:
            env_info['is_server'] = True
            break

    # Check if it's likely a server (many cores, no display)
    if env_info['cpu_count'] >= 16:
        env_info['is_server'] = True

    # Check for display (GUI) - local PCs typically have one
    if 'DISPLAY' in os.environ or platform.system() == 'Windows':
        env_info['is_local'] = True
    elif env_info['cpu_count'] <= 8:
        env_info['is_local'] = True

    # Check hostname patterns
    hostname = socket.gethostname().lower()
    if any(x in hostname for x in ['node', 'compute', 'server', 'cluster', 'vm']):
        env_info['is_server'] = True

    return env_info
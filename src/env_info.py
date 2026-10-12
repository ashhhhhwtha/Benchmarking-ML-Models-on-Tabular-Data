import platform, subprocess, sys

def get_env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    for pkg in ["numpy", "pandas", "sklearn", "xgboost", "torch"]:
        try:
            info[pkg] = __import__(pkg).__version__
        except ImportError:
            pass
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        info["git_commit"] = None
    return info
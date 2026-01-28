#!/usr/bin/env python3
"""
Remote deploy and launch training on a server via SSH (Paramiko).

Usage:
  python scripts/remote_deploy_and_launch.py \
    --host i-2.gpushare.com --port 23937 --user root --password '***' \
    --remote-dir /root/RP1 --launch structural --seeds 30 --timesteps 100000 --eval-episodes 50 --load-multiplier 5.0

  python scripts/remote_deploy_and_launch.py \
    --host i-2.gpushare.com --port 23937 --user root --password '***' \
    --remote-dir /root/RP1 --launch capacity --shape uniform --capacities 10,30 --loads 5 --algos A2C,PPO --seeds 5 --timesteps 100000 --eval-episodes 50

This script:
  - uploads a tarball containing Code/, scripts/, Analysis/statistical_analysis/, README.md
  - sets up a Python venv in <remote-dir>/.venv
  - installs Python dependencies
  - launches the requested experiment under nohup and writes PID/log files in <remote-dir>/logs
"""

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path
import paramiko


def create_tarball(output_path: Path):
    root = Path(__file__).resolve().parents[1]
    include = [root / 'Code', root / 'scripts', root / 'Analysis' / 'statistical_analysis', root / 'README.md']
    with tarfile.open(output_path, 'w:gz') as tar:
        for item in include:
            if item.exists():
                arcname = item.relative_to(root)
                tar.add(item, arcname=str(arcname))


def ssh_connect(host: str, port: int, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=user, password=password, timeout=30)
    return client


def run_cmd(ssh: paramiko.SSHClient, cmd: str):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def ensure_venv(ssh: paramiko.SSHClient, remote_dir: str) -> str:
    """Ensure Python available; try venv, fallback to system python.
    Returns the python binary path to use for launches.
    """
    # Always ensure directories
    rc, out, err = run_cmd(ssh, f"mkdir -p {remote_dir} {remote_dir}/logs")
    if rc != 0:
        raise RuntimeError(err)

    # Ensure python3 and pip
    rc, out, err = run_cmd(ssh, "command -v python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y python3 python3-pip)")
    # Try venv path
    venv_py = f"{remote_dir}/.venv/bin/python"
    rc, out, err = run_cmd(ssh, f"python3 -m venv {remote_dir}/.venv")
    if rc != 0:
        # Attempt to install python3-venv and re-try
        run_cmd(ssh, "apt-get update && apt-get install -y python3-venv >/dev/null 2>&1 || true")
        run_cmd(ssh, "python3 -m ensurepip --upgrade >/dev/null 2>&1 || true")
        rc, out, err = run_cmd(ssh, f"python3 -m venv {remote_dir}/.venv")

    if rc == 0:
        # Use venv and install deps
        cmds = [
            f"{venv_py} -m pip install --upgrade pip setuptools wheel",
            f"{venv_py} -m pip install 'numpy<2.0' 'gymnasium>=0.29,<1.0' 'stable-baselines3>=2.2,<3' 'torch>=2.1,<2.3' pandas scipy matplotlib seaborn tensorboard"
        ]
        for c in cmds:
            r2, o2, e2 = run_cmd(ssh, c)
            if r2 != 0:
                raise RuntimeError(f"Command failed: {c}\n{o2}\n{e2}")
        return venv_py
    else:
        # Fall back to system python3
        # Install deps globally
        cmds = [
            "python3 -m pip install --upgrade pip setuptools wheel",
            "python3 -m pip install 'numpy<2.0' 'gymnasium>=0.29,<1.0' 'stable-baselines3>=2.2,<3' 'torch>=2.1,<2.3' pandas scipy matplotlib seaborn tensorboard"
        ]
        for c in cmds:
            r2, o2, e2 = run_cmd(ssh, c)
            if r2 != 0:
                raise RuntimeError(f"Command failed: {c}\n{o2}\n{e2}")
        return "python3"


def upload_tar(ssh: paramiko.SSHClient, local_tar: Path, remote_dir: str):
    sftp = ssh.open_sftp()
    remote_tar = f"{remote_dir}/rp1_upload.tgz"
    sftp.put(str(local_tar), remote_tar)
    sftp.close()
    rc, out, err = run_cmd(ssh, f"tar xzf {remote_tar} -C {remote_dir} && rm -f {remote_tar}")
    if rc != 0:
        raise RuntimeError(f"Extract failed: {out}\n{err}")


def launch_experiment(ssh: paramiko.SSHClient, remote_dir: str, args: argparse.Namespace, py_bin: str):
    py = py_bin
    logs = f"{remote_dir}/logs"
    if args.launch == 'structural':
        cmd = (
            f"nohup {py} {remote_dir}/Code/training_scripts/run_structural_comparison_5x_load.py "
            f"--mode all --n-seeds {args.seeds} --timesteps {args.timesteps} "
            f"--eval-episodes {args.eval_episodes} --load-multiplier {args.load_multiplier} "
            f">> {logs}/structural_5x_{args.seeds}s_{args.timesteps}t_{args.eval_episodes}e.log 2>&1 & echo $! > {logs}/structural_5x.pid"
        )
    elif args.launch == 'capacity':
        cmd = (
            f"nohup {py} {remote_dir}/Code/training_scripts/run_capacity_scan.py "
            f"--include-heuristics --capacities {args.capacities} --loads {args.loads} --shape {args.shape} "
            f"--algos {args.algos} --n-seeds {args.seeds} --timesteps {args.timesteps} --eval-episodes {args.eval_episodes} "
            f">> {logs}/capacity_{args.shape}_{args.seeds}s_{args.timesteps}t_{args.eval_episodes}e.log 2>&1 & echo $! > {logs}/capacity_{args.shape}.pid"
        )
    else:
        raise ValueError("Unknown launch mode")
    rc, out, err = run_cmd(ssh, cmd)
    if rc != 0:
        raise RuntimeError(f"Launch failed: {out}\n{err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', required=True)
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--user', required=True)
    p.add_argument('--password', required=True)
    p.add_argument('--remote-dir', default='/root/RP1')
    p.add_argument('--launch', choices=['structural','capacity'], required=True)
    # shared
    p.add_argument('--seeds', type=int, default=30)
    p.add_argument('--timesteps', type=int, default=100000)
    p.add_argument('--eval-episodes', type=int, default=50)
    # structural
    p.add_argument('--load-multiplier', type=float, default=5.0)
    # capacity
    p.add_argument('--shape', default='uniform')
    p.add_argument('--capacities', default='10,30')
    p.add_argument('--loads', default='5')
    p.add_argument('--algos', default='A2C,PPO')
    args = p.parse_args()

    # Create tarball
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / 'rp1_upload.tgz'
        create_tarball(tar_path)

        # SSH
        ssh = ssh_connect(args.host, args.port, args.user, args.password)
        try:
            # Create remote dir and upload code
            rc, out, err = run_cmd(ssh, f"mkdir -p {args.remote_dir}")
            if rc != 0:
                raise RuntimeError(err)
            upload_tar(ssh, tar_path, args.remote_dir)
            # Ensure venv and deps
            py_bin = ensure_venv(ssh, args.remote_dir)
            # Launch job
            launch_experiment(ssh, args.remote_dir, args, py_bin)
            print("✅ Launched successfully. Logs in:", f"{args.remote_dir}/logs")
        finally:
            ssh.close()


if __name__ == '__main__':
    main()

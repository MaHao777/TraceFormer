"""临时 SSH 会话与传输；密码无回显输入，不保存、不修改登录授权。"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1] / "outputs/paper_cuda_20260928/ssh"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="connect.cqa1.seetacloud.com")
    parser.add_argument("--port", type=int, default=35631)
    parser.add_argument("--session-root", type=Path, default=ROOT)
    parser.add_argument("--remote-root", default="/root/autodl-tmp/traceformer-paper-20260928")
    args = parser.parse_args()
    args.session_root.mkdir(parents=True, exist_ok=True)
    known = args.session_root / "known_hosts"
    client = paramiko.SSHClient()
    if known.exists():
        client.load_host_keys(str(known))
    # 首次固定主机密钥，此后拒绝密钥变更；不修改远端登录授权。
    client.set_missing_host_key_policy(paramiko.RejectPolicy() if known.exists() else paramiko.AutoAddPolicy())
    password = getpass.getpass("SSH password (not stored): ")
    try:
        client.connect(args.host, port=args.port, username="root",
                       timeout=25, auth_timeout=25, banner_timeout=25,
                       allow_agent=False, look_for_keys=False, password=password)
        password = None
        client.save_host_keys(str(known))
        host_key = client.get_transport().get_remote_server_key()
        fingerprint = base64.b64encode(hashlib.sha256(host_key.asbytes()).digest()).decode().rstrip("=")
        metadata = {"host": args.host, "port": args.port,
                    "user": "root", "host_fingerprint": f"SHA256:{fingerprint}",
                    "remote_root": args.remote_root}
        (args.session_root / "connection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps({"ready": metadata}), flush=True)
        # 标准输入只接收 JSON 操作；连接密码不写入文件或日志。
        for line in sys.stdin:
            request = json.loads(line)
            action = request["action"]
            if action == "close":
                break
            if action == "exec":
                _, out, err = client.exec_command(request["command"])
                print(out.read().decode(errors="replace"), end="", flush=True)
                print(err.read().decode(errors="replace"), end="", flush=True)
                print(json.dumps({"exit_code": out.channel.recv_exit_status()}), flush=True)
            elif action in {"put", "get"}:
                with client.open_sftp() as sftp:
                    if action == "put":
                        sftp.put(request["source"], request["destination"])
                    else:
                        Path(request["destination"]).parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(request["source"], request["destination"])
                print(json.dumps({"transfer": "complete", "action": action}), flush=True)
            else:
                raise ValueError("unknown SSH operation")
    finally:
        password = None
        client.close()

if __name__ == "__main__":
    main()

import json, shlex, subprocess, sys, os

def build_cmd(cfg: dict) -> list[str]:
    # Accelerate prefix
    accel = cfg.get("accelerate", {})
    accel_args = accel.get("args", [])
    if not isinstance(accel_args, list):
        raise ValueError("accelerate.args must be a list")

    entrypoint = cfg.get("entrypoint", "sdxl_train_network.py")

    cmd = ["accelerate", "launch", *accel_args, entrypoint]

    # Map remaining keys to --k v (booleans become flags, lists repeat)
    skip_keys = {"accelerate", "entrypoint"}
    for k, v in cfg.items():
        if k in skip_keys:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif isinstance(v, list):
            for item in v:
                cmd.extend([flag, str(item)])
        else:
            cmd.extend([flag, str(v)])
    return cmd

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(json_path, "r") as f:
        cfg = json.load(f)

    cmd = build_cmd(cfg)
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    # Inherit env; prints live logs
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

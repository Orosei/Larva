import subprocess
import sys


def run_command(cmd):
    """运行命令并返回结果"""
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout, result.stderr, result.returncode


def check_pip_mirror():
    """检查 pip 当前镜像设置"""
    out, err, rc = run_command("pip config list")
    if "global.index-url" in out:
        print("✅ pip 已配置镜像:", out.strip())
        return True
    else:
        print("⚠️ pip 未配置镜像，使用默认源")
        return False


def set_pip_mirror():
    """设置 pip 为清华镜像"""
    cmd = "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"
    out, err, rc = run_command(cmd)
    if rc == 0:
        print("✅ pip 镜像已设置为清华源")
    else:
        print("❌ pip 镜像设置失败:", err)


def check_conda_mirror():
    """检查 Conda 当前镜像设置"""
    out, err, rc = run_command("conda config --get channels")
    if "mirrors.tuna.tsinghua.edu.cn" in out:
        print("✅ Conda 已配置清华镜像")
        return True
    else:
        print("⚠️ Conda 未配置镜像，使用默认源")
        return False


def set_conda_mirror():
    """设置 Conda 为清华镜像"""
    cmds = [
        "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
        "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r",
        "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2",
        "conda config --set show_channel_urls yes",
    ]
    for cmd in cmds:
        out, err, rc = run_command(cmd)
        if rc != 0:
            print(f"❌ 执行命令失败: {cmd}\n错误信息: {err}")
            return
    print("✅ Conda 镜像已设置为清华源")


def main():
    print("🔍 正在检测当前 pip/Conda 镜像设置...\n")

    # 检查并设置 pip 镜像
    if not check_pip_mirror():
        set_pip_mirror()

    print()

    # 检查并设置 Conda 镜像
    if not check_conda_mirror():
        set_conda_mirror()

    print("\n🎉 镜像配置完成！以后 pip 和 Conda 安装包会自动走国内源，速度更快。")


if __name__ == "__main__":
    main()

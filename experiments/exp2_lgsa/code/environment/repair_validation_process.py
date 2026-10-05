"""Separate installed-runtime checks from the installer's pip imports."""
import ast
from pathlib import Path
import textwrap

root = Path(__file__).resolve().parents[1]
installer = root / 'environment/install_all_cpu.py'
checker = root / 'environment/check_cpu.py'
source = installer.read_text()
replacement = "        run([sys.executable, '-u', info / 'check_cpu.py'])\n"
start_marker = '        import torch\n'
end_marker = "        print('LGSA_ENV_CPU_READY', flush=True)\n"

if replacement in source:
    assert checker.is_file(), 'Standalone checker missing'
    ast.parse(checker.read_text())
    print('VALIDATION_PROCESS_FIX_ALREADY_APPLIED')
else:
    assert source.count(start_marker) == source.count(end_marker) == 1, 'Unexpected installer layout'
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    body = textwrap.dedent(source[start:end])
    header = '''import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path('/gpfs/work/aac/zixinfeng23/fyp/LGSA-GoEmotions')
ENV = Path('/gpfs/work/aac/zixinfeng23/.conda/envs/lgsa_goemo')
assert Path(sys.prefix).resolve() == ENV.resolve(), 'Wrong Python environment'
assert sys.version_info[:2] == (3, 10)
os.chdir(ROOT)
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)
info = ROOT / 'environment'
upstream = ROOT / 'original_lgsa'
pip = [sys.executable, '-m', 'pip']
head = run(['git', '-C', upstream, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
assert head == '4ea3dfb39dcee82654f443811471dd29be32e87e'
assert not run(['git', '-C', upstream, 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
fingerprint = (info / 'dependency-plan.fingerprint').read_text().strip()
run(pip + ['check'])
print('Checking installed packages in a fresh Python process...', flush=True)
'''
    updated = source[:start] + replacement + source[end:]
    ast.parse(header + body)
    ast.parse(updated)
    backup = installer.with_name(installer.name + '.before-check-process-fix')
    if not backup.exists():
        with backup.open('x') as stream:
            stream.write(source)
    checker.write_text(header + body)
    temporary = installer.with_name(installer.name + '.check-process.tmp')
    temporary.write_text(updated)
    temporary.replace(installer)
    print('VALIDATION_PROCESS_FIX_APPLIED')

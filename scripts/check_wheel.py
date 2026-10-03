"""Build and install the actual wheel without runtime dependencies; check durable."""
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
from poetry.core.masonry.api import build_wheel

def main():
    with tempfile.TemporaryDirectory(prefix='epoch-wheel-') as work:
        root=Path(work)
        wheel=root/build_wheel(str(root))
        environment=root/'environment'
        venv.EnvBuilder(with_pip=True).create(environment)
        executable=environment/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        subprocess.run([str(executable),'-m','pip','install','--no-index','--no-deps',str(wheel)],check=True)
        probe="import durable, pathlib, sys; assert pathlib.Path(durable.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)); from durable.store import Store; print('Installed durable:', durable.__file__)"
        subprocess.run([str(executable),'-I','-c',probe],cwd=root,check=True)
        subprocess.run([str(executable),'-I','-m','durable.experiment','--help'],cwd=root,check=True)
        print('Wheel build, isolated installation, durable imports and experiment CLI passed.')
if __name__=='__main__':
    main()

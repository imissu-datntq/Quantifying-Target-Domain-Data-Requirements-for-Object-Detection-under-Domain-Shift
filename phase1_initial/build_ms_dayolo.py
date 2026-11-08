"""Build pinned MS-DAYOLO with workspace-local CUDA and OpenCV dependencies."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    prefix = root / '.venvs/ms-dayolo'
    source = root / 'external/MS-DAYOLO'
    sha = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != 'c655ceed3b99e6b499a9c907a3ae9c2deba98c1e':
        raise RuntimeError('Unexpected MS-DAYOLO revision')
    env = os.environ.copy()
    env['PATH'] = str(prefix / 'bin') + os.pathsep + env['PATH']
    env['PKG_CONFIG_LIBDIR'] = str(prefix / 'lib/pkgconfig')
    env['LD_LIBRARY_PATH'] = str(prefix / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    compiler = str(prefix / 'bin/x86_64-conda-linux-gnu-')
    cflags = subprocess.check_output(['pkg-config', '--cflags', 'opencv4'], env=env, text=True).strip()
    libraries = subprocess.check_output(['pkg-config', '--libs', 'opencv4'], env=env, text=True).strip()
    common = f'-Iinclude -I3rdparty/stb/include -I{prefix}/include -DGPU -DCUDNN -DCUDNN_HALF -DOPENCV {cflags}'
    ldflags = f'-L{prefix}/lib -L{prefix}/lib/stubs -Wl,-rpath,{prefix}/lib -lm -pthread -lstdc++ -lcuda -lcudart -lcublas -lcurand -lcudnn {libraries}'
    command = ['make', f'-j{args.jobs}', 'darknet', f'CC={compiler}gcc', f'CPP={compiler}g++',
               f'NVCC={prefix}/bin/nvcc -ccbin {compiler}g++',
               'ARCH=-gencode arch=compute_86,code=[sm_86,compute_86]',
               f'COMMON={common}', f'LDFLAGS={ldflags}',
               'CFLAGS=-O3 -Wall -Wno-unused-result -fPIC -DGPU -DCUDNN -DCUDNN_HALF -DOPENCV']
    print(json.dumps(dict(cwd=str(source), command=command, upstream_sha=sha), indent=2), flush=True)
    if not args.dry_run:
        (source / 'obj').mkdir(exist_ok=True)
        subprocess.run(command, cwd=source, env=env, check=True)


if __name__ == '__main__':
    main()

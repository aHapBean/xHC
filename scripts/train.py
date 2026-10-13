"""Launch a YAML training recipe with torchrun."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from string import Template
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
MEGATRON = ROOT / 'Megatron-LM'


def render(config, environment):
    execution = config['execution']
    values = dict(environment)
    values.update(VARIANT=config['variant'], HC_SETTINGS=json.dumps(config['hyper_connection_settings']))
    values.setdefault('MBS', str(execution['micro_batch_size']))
    values.setdefault('EP', str(execution['expert_model_parallel_size']))
    for key in ('MBS', 'EP'):
        if int(values[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    result = []
    for key, value in config['arguments'].items():
        if value is False or value is None:
            continue
        result.append('--' + key)
        if value is not True:
            result.append(Template(str(value)).substitute(values))
    dispatcher = values.get('MOE_DISPATCHER', execution['dispatcher'])
    if dispatcher == 'alltoall':
        result += ['--moe-token-dispatcher-type', 'alltoall']
    elif dispatcher in ('deepep', 'hybridep'):
        result += ['--moe-token-dispatcher-type', 'flex', '--moe-flex-dispatcher-backend', dispatcher]
        if dispatcher == 'hybridep':
            result += ['--moe-hybridep-pad-multiple', str(execution['hybridep_pad_multiple'])]
    else:
        raise ValueError(f'Unknown dispatcher: {dispatcher}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--backend', choices=('reference', 'fused_xhc'), default='reference')
    parser.add_argument('--dry-run', action='store_true')
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ['--']:
        overrides = overrides[1:]
    config = yaml.safe_load(args.config.read_text())
    if args.backend == 'fused_xhc' and config['variant'] != 'xhc':
        parser.error('fused_xhc requires an xHC recipe')
    env = dict(os.environ)
    for key in ('DATA_ARGS', 'TOKENIZER_DIR'):
        if not env.get(key):
            parser.error(f'Set {key} to your local data/tokenizer path')
    if not Path(env['DATA_ARGS']).is_file() or not Path(env['TOKENIZER_DIR']).is_dir():
        parser.error('DATA_ARGS must be a file and TOKENIZER_DIR must be a directory')
    outputs = Path(env.get('EXPERIMENTS_ROOT', ROOT / 'experiments')).resolve()
    name = env.get('EXP_NAME', args.config.stem + ('_efficient' if args.backend == 'fused_xhc' else ''))
    env.setdefault('RUN_DIR', str(outputs / name))
    env.setdefault('TENSORBOARD_DIR', str(outputs / 'tensorboard' / name))
    env.setdefault('DATA_CACHE', str(outputs / 'data_cache'))
    env['PYTHONPATH'] = str(MEGATRON) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env.setdefault('OMP_NUM_THREADS', '1')
    env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    env['CUDA_HOME'] = env.get('CUDA_HOME') or '/usr/local/cuda'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    training_args = render(config, env) + ['--hyper-connection-backend', args.backend]
    if env.get('ACTIVATION_STATS', '1') == '1':
        training_args.append('--log-activation-stats')
    cuda = Path(env['CUDA_HOME'])
    if (cuda / 'bin/ptxas').is_file():
        env.setdefault('TRITON_PTXAS_PATH', str(cuda / 'bin/ptxas'))
        env['CPATH'] = str(cuda / 'include') + (os.pathsep + env['CPATH'] if env.get('CPATH') else '')
    training_args += overrides
    nodes, gpus = int(env.get('NNODES', '1')), int(env.get('GPUS', '4'))
    rank = int(env.get('NODE_RANK', '0'))
    if nodes < 1 or gpus < 1 or not 0 <= rank < nodes:
        parser.error('Require NNODES/GPUS > 0 and 0 <= NODE_RANK < NNODES')
    distributed = ['--standalone']
    if nodes > 1:
        if not env.get('MASTER_ADDR'):
            parser.error('Set MASTER_ADDR for multi-node training')
        distributed = [f'--nnodes={nodes}', f'--node_rank={rank}',
                       '--master_addr=' + env['MASTER_ADDR'],
                       '--master_port=' + env.get('MASTER_PORT', '29500')]
    command = [sys.executable, '-m', 'torch.distributed.run', *distributed,
               f'--nproc_per_node={gpus}', str(MEGATRON / 'pretrain_gpt.py'), *training_args]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    for key in ('RUN_DIR', 'TENSORBOARD_DIR', 'DATA_CACHE'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record = Path(env['RUN_DIR']) / 'configs' / stamp
        record.mkdir(parents=True)
        (record / 'recipe.yaml').write_text(args.config.read_text())
        (record / 'arguments.json').write_text(json.dumps(training_args, indent=2) + '\n')
    # Dataset helper compilation is serialized across nodes sharing this checkout.
    import fcntl
    with (MEGATRON / 'megatron/core/datasets/Makefile').open() as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        subprocess.run(['make', '-C', str(MEGATRON / 'megatron/core/datasets')], env=env, check=True)
    os.chdir(MEGATRON)
    os.execvpe(sys.executable, command, env)


if __name__ == '__main__':
    main()

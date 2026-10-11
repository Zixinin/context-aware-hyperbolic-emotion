"""Run lightweight CPU tests and save a source-bound machine-readable result."""
import argparse
from pathlib import Path
import platform
import unittest
from protocol import HERE, package_hashes, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(HERE / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'status': 'PEER_UNIT_TESTS_PASS' if result.wasSuccessful() else 'FAILED',
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'system': platform.system(), 'hostname': platform.node(),
              'package_sha256': package_hashes(), 'scope': 'CPU unit/toy adapter tests; not real LGSA regression'}
    write(args.out, record)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

import argparse
import os
from datetime import datetime

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB

from ConfigSpace.read_and_write import pcs_new, json

from hpbandster_helpers import get_configspace_and_basecmd, PatchWorker

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default="config_files/config_standard_bs1.json", help="Directory for JSON config file")
args = parser.parse_args()

# Worker test
# base_cmd, config_space = get_configspace_and_basecmd(args.config_file)
# now = datetime.now()
# worker = PatchWorker(run_id=now.astimezone().tzinfo.tzname(None) + now.strftime('%Y%m%d_%H_%M_%S_%f'),
#                      base_cmd=base_cmd)
# conf = config_space.sample_configuration().get_dictionary()
# print(worker.compute(config=conf, budget=1, working_directory="./"))
#

hpo_dir = './'
num_workers = 1
min_epoch = 1
max_epoch = 5
num_iter = 3
eta = 2 # some sort of param for BOHB

nameserver = hpns.NameServer(run_id='local_run', nic_name=None, working_directory=hpo_dir)
host, port = nameserver.start()
workers = []
base_cmd, config_space = get_configspace_and_basecmd(args.config_file)
for i in range(num_workers):
  worker = PatchWorker(run_id='local_run',
                       base_cmd=base_cmd,
                       nameserver=host,
                       nameserver_port=port)
  worker.run(background=True)
  print(f"Worker {i} Running")
  workers.append(worker)
result_logger = hpres.json_result_logger(directory=hpo_dir, overwrite=True)
with open(os.path.join(hpo_dir, 'configspace.json'), 'w') as f:
  f.write(json.write(config_space))

optimizer = BOHB(configspace=config_space,
                 run_id="local_run",
                 min_budget=min_epoch,
                 max_budget=max_epoch,
                 eta=eta,
                 host=host,
                 nameserver=host,
                 nameserver_port=port,
                 result_logger=result_logger
                 )
print(f"BOHB Running")
result = optimizer.run(n_iterations=num_iter, min_n_workers=num_workers)

optimizer.shutdown(shutdown_workers=True)
nameserver.shutdown()

with open(os.path.join(hpo_dir, 'results.pkl'), 'wb') as f:
  import pickle
  pickle.dump(result, f)

id2config = result.get_id2config_mapping()
incumbent = result.get_incumbent_id()
inc_value = result.get_runs_by_id(incumbent)[-1]['loss']
inc_cfg = id2config[incumbent]['config']

# to develop a full blown visual analysis, see https://github.com/automl/BOAH/blob/master/examples/mlp_on_digits/notebook.ipynb

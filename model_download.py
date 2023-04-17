import os
import sys

from determined.experimental import client
from determined import pytorch

checkpoint = client.get_experiment(179) \
    .top_checkpoint(sort_by='validation_loss',
                    smaller_is_better=True)
path = checkpoint.download()

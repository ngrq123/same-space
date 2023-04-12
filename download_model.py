import sys

from determined.experimental import client

print('Downloading model', sys.argv[1])
checkpoint = client.get_checkpoint(sys.argv[1])
checkpoint.download()

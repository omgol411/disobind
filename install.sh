#!/bin/bash

# Unzip ProtTrans dir.
tar -xzvf ./ProtTrans.tar.gz

# Export disobind path.
if [ -z "$PYTHONPATH" ]; then
    echo "export PYTHONPATH=`pwd`:$PYTHONPATH" >> ~/.bash_profile
else
    echo "export PYTHONPATH=$PYTHONPATH:`pwd`" >> ~/.bash_profile
fi
source ~/.bash_profile

# Install dependencies.
pip install ./requirements.txt
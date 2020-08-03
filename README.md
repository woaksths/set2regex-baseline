# set2regex-baseline
Generating regular expressions from both natural language and examples

## Dataset download
> [baseline dataset](https://github.com/woaksths/set2regex_dataset "set2regex dataset link")
 

## Install 
    $python3 -m venv venv
    $source venv/bin/activate
    $pip install -r requirements.txt
    $python setup.py install
    $git clone https://github.com/0xnurl/fado-python3.git
    $python fado-python3/setup.py install
    
    
## Usage
    # Before running this command, check the training option via $python examples/sample.py --help
    $python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_DIR
    $python examples/evaluation.py --train_path $TRAIN_PATH --test_path $TEST_PATH --checkpoint $CHECKPOINT
    

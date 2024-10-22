SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

virtualenv venv
source $SCRIPT_DIR/venv/bin/activate
pip install -r requirements.txt

conda env list
conda create --name myenv python=3.5
source activate myenv
source deactivate
conda env remove --name myenv

pip freeze > requirements.txt
pip install -r requirements.txt

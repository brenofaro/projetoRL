## Windowns
### Baixar e instalar o miniconda
1. Baixar o miniconda no site [Miniconda Install](https://docs.anaconda.com/miniconda/install/)

### Preparar ambiente conda
1. Criar um ambiente conda com o nome de `env`
```bash
conda create --name env python=3.11
```
2. Ativar o ambiente conda
```bash
conda activate env
```
3. Instalar o pytorch
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

4. Instalar outras dependências
```bash
pip install torch torchvision gymnasium[atari] numpy scikit-image torchsummary ale_py
```

### Recomendado para desenvolvimento
1. Instalar o jupyter notebook
```bash
pip install jupyter
```
2. Instalar ipykernel
```bash
pip install ipykernel
```
3. Adicionar o ambiente kernel ao jupyter notebook
```bash
python -m ipykernel install --user --name=env
```

### Execução do código
1. Clonar o repositório
```bash
git clone
```
2. Acessar a pasta do projeto com o vscode
```bash
code .
```
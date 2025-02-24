# Projeto de Aprendizado por Reforço com Pytorch
Projeto de aprendizado por reforço utilizando o pytorch para treinar um agente para jogar o jogo de atari space invaders.

Feito usando os conceitos de DQN e CNN.

## Windowns
### Baixar e instalar o miniconda
1. Baixar o miniconda (QuickStart install) no site [Miniconda Install](https://docs.anaconda.com/miniconda/install/)

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
pip install torch torchvision gymnasium[atari] numpy scikit-image torchsummary ale_py tensorboard
```

5. Instalar o jupyter notebook
```bash
pip install jupyter
```
6. Instalar ipykernel
```bash
pip install ipykernel
```
7. Adicionar o ambiente kernel ao jupyter notebook
```bash
python -m ipykernel install --user --name=env
```

### Execução do código
1. Clonar o repositório
```bash
git clone https://github.com/brenofaro/projetoRL.git
```
2. Acessar a pasta do projeto com o vscode
```bash
code .
```
3. Instalar as extensões do vscode
- [Python](https://marketplace.visualstudio.com/items?itemName=donjayamanne.python-extension-pack)
- [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

4. Selecionar o kernel do ambiente conda

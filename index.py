# Comandos para criar repositório e publicar arquivos usando GitHub CLI (gh):

# 1. Inicialize o git na pasta do projeto:
#    git init

# 2. Adicione todos os arquivos:
#    git add .

# 3. Faça o primeiro commit:
#    git commit -m "primeiro commit"

# 4. Crie o repositório no GitHub usando o gh CLI:
#    gh repo create nome-do-repositorio --public --source=. --remote=origin --push

# 5. Se já tiver o repositório criado, apenas conecte e envie:
#    git remote add origin https://github.com/seu-usuario/nome-do-repositorio.git
#    git push -u origin main

# Observação: Instale o GitHub CLI se necessário: https://cli.github.com/
# Autentique-se com: gh auth login

import os
import subprocess
import shutil
import stat

# Solicita o nome do repositório ao usuário
repo_name = input("Digite o nome do repositório: ")
github_user = "domzack"
repo_path = f"c:/mygits/{repo_name}"


def on_rm_error(func, path, exc_info):
    # Tenta alterar permissões e remover novamente
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"Falha ao remover '{path}': {e}")


# Verifica se a pasta já existe
if os.path.exists(repo_path):
    resp = input(
        f"A pasta '{repo_path}' já existe. Deseja remover e criar uma nova? (s/n): "
    ).lower()
    if resp == "s":
        try:
            shutil.rmtree(repo_path, onerror=on_rm_error)
        except Exception as e:
            print(f"Erro ao remover a pasta: {e}")
            print(
                "Feche arquivos ou processos que estejam usando a pasta e tente novamente."
            )
            exit()
    else:
        print("Operação cancelada pelo usuário.")
        exit()

# Cria a pasta do repositório
os.makedirs(repo_path, exist_ok=True)

# Acessa a pasta
os.chdir(repo_path)

# Inicializa o git
subprocess.run(["git", "init"])

# Cria o repositório no GitHub
subprocess.run(
    [
        "gh",
        "repo",
        "create",
        f"{github_user}/{repo_name}",
        "--public",
        "--source=.",
        "--remote=origin",
        "--description",
        f"Repositório {repo_name}",
    ]
)

# Cria e troca para o branch main
subprocess.run(["git", "checkout", "-b", "main"])

# Adiciona todos os arquivos
subprocess.run(["git", "add", "."])

# Verifica se há arquivos para commit
result = subprocess.run(
    ["git", "status", "--porcelain"], capture_output=True, text=True
)
if not result.stdout.strip():
    print("Nenhum arquivo para commit. Criando README.md...")
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("helloworld\n")
    subprocess.run(["git", "add", "README.md"])

# Faz o commit inicial
subprocess.run(["git", "commit", "-m", f"Commit inicial do projeto {repo_name}"])

# Verifica se o remoto 'origin' existe
remotes = subprocess.run(["git", "remote"], capture_output=True, text=True)
if "origin" not in remotes.stdout.split():
    print("O repositório remoto 'origin' não foi criado corretamente.")
    print(
        "Verifique se o comando 'gh repo create' foi executado com sucesso e se você tem acesso ao GitHub CLI."
    )
    exit()

# Publica os arquivos no GitHub
subprocess.run(["git", "push", "-u", "origin", "main"])

# Exibe a URL do repositório
print(
    f"Repositório criado e publicado em: https://github.com/{github_user}/{repo_name}"
)

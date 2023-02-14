export ZSH="/home/vscode/.oh-my-zsh"
ZSH_THEME="cloud"
plugins=(
    git
    zsh-autosuggestions
    zsh-syntax-highlighting
)
source $ZSH/oh-my-zsh.sh
alias ll='ls -la'
alias tgp='terragrunt plan'
alias tga='terragrunt apply -auto-approve'
alias tgi='terragrunt init'
alias tgpo='terragrunt plan -no-color > plan.txt'

function validate-all() {
    folder="staging"
    if [ "$1" ]; then
        folder="$1"
    fi
    cd /workspaces/hnry-terraform/$folder
    terragrunt run-all validate
}
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=output.log

export PATH="$HOME/ollama/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11500
unset OLLAMA_MODELS            # or set it to a real folder you own

ollama serve > ollama.log 2>&1 &
sleep 10                        # give the server time to start
ollama pull qwen2.5:3b          # no-op if it's already downloaded

cd ~/code/agents/Research_assistant
python orchestrator.py

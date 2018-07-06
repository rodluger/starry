import subprocess
hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")[:-1]
with open("gitlinks.tex", "w") as f:
    print(r"\newcommand{\codelink}[1]{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.py}{\codeicon}\,\,}" % hash, file=f)
    print(r"\newcommand{\animlink}[1]{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.gif}{\animicon}\,\,}" % hash, file=f)
    print(r"\newcommand{\prooflink}[1]{\href{https://github.com/rodluger/starry/blob/%s/docs/proofs/#1.ipynb}{\raisebox{-0.1em}{\prooficon}}}" % hash, file=f)

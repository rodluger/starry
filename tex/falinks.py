import subprocess
hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")[:-1]
with open(".codelink.tex", "w") as f:
    print(r"\newcommand{\codelink}[1]{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.py}{\color{linkcolor}\faCogs}\,\,}" % hash, file=f)
with open(".animlink.tex", "w") as f:
    print(r"\newcommand{\animlink}[1]{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.gif}{\color{linkcolor}\faPlayCircle}\,\,}" % hash, file=f)
with open(".prooflink.tex", "w") as f:
    print(r"\newcommand{\prooflink}[1]{\href{https://github.com/rodluger/starry/blob/%s/tex/notebooks/#1.ipynb}{\raisebox{-0.1em}{\color{linkcolor}\faPencilSquareO}}}" % hash, file=f)

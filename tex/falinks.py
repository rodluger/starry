import subprocess
hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")[:-1]
with open(".code.tex", "w") as f:
    print(r"\newcommand{\code}[1]{\marginnote{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.py}{\color{linkcolor}\Large\faCogs}}}" % hash, file=f)
with open(".proof.tex", "w") as f:
    print(r"\newcommand{\proof}[1]{\marginnote{\href{https://github.com/rodluger/starry/blob/%s/tex/notebooks/#1.ipynb}{\color{linkcolor}\Large\faPencilSquareO}}}" % hash, file=f)
with open(".animation.tex", "w") as f:
    print(r"\newcommand{\animation}[1]{\marginnote{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.gif}{\color{linkcolor}\Large\faPlayCircle}}}" % hash, file=f)
with open(".figanimation.tex", "w") as f:
    print(r"\newcommand{\figanimation}[1]{\marginnote{\hspace{1.95em}\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.gif}{\color{linkcolor}\Large\faPlayCircle}}}" % hash, file=f)

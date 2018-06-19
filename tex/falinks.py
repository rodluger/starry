import subprocess
hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")[:-1]
with open(".code.tex", "w") as f:
    print(r"\newcommand{\code}[1]{\marginnote{\href{https://github.com/rodluger/starry/blob/%s/tex/figures/#1.py}{\color{linkcolor}\Large\faFileCodeO}}}" % hash, file=f)
with open(".proof.tex", "w") as f:
    print(r"\newcommand{\proof}[1]{\marginnote{\href{https://github.com/rodluger/starry/raw/%s/tex/notebooks/#1.ipynb}{\color{linkcolor}\Large\faPencilSquareO}}}" % hash, file=f)
with open(".animation.tex", "w") as f:
    print(r"\newcommand{\animation}[1]{\marginnote{\href{https://github.com/rodluger/starry/tree/%s/tex/figures/#1.gif}{\color{linkcolor}\Large\faPlayCircleO}}}" % hash, file=f)
with open(".figanimation.tex", "w") as f:
    print(r"\newcommand{\figanimation}[1]{\marginnote{\hspace{1.75em}\href{https://github.com/rodluger/starry/tree/%s/tex/figures/#1.gif}{\color{linkcolor}\Large\faPlayCircleO}}}" % hash, file=f)

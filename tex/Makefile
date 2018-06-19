.PHONY : clean, default, publish

TEX_FILES   = starry
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun starry.log
FIGURES     = $(patsubst %.py,%.pdf,$(wildcard figures/*.py))
SOURCES     = $(wildcard ../src/*.[ch])

default: starry.pdf

starry.pdf: starry.tex stylez.tex $(FIGURES) $(SOURCES)
	python falinks.py
	${LATEX} starry.tex
	bibtex starry
	( ${CHECK_RERUN} && ${LATEX} starry.tex ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} starry.tex ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} starry.tex ) || echo "Done."

clean:
	$(RM_TMP)
	rm -rf starryNotes.bib
	rm -rf stability_test*pdf
	rm -rf $(FIGURES)

figures/%.pdf: figures/%.py
	#cd $(<D);python $(<F)

publish:
	cd ..; ./publish.sh

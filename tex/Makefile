.PHONY : clean, default, publish

TEX_FILES   = starry
TMP_SUFFS   = pdf aux bbl blg log dvi ps eps out ent
LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
SUFF        = pdf
RM_TMP      = $(foreach d, ${TEX_FILES}, rm -rf $(foreach suff, ${TMP_SUFFS}, ${d}.${suff}))
CHECK_RERUN = grep Rerun starry.log
FIGURES     = $(patsubst %.py,%.pdf,$(wildcard figures/*.py))
SOURCES     = $(wildcard ../src/*.[ch])
TECTONIC    = $(shell command -v tectonic >/dev/null && echo true || echo false )

default: starry.pdf

starry.pdf: starry.tex stylez.tex starry.bib $(FIGURES) $(SOURCES)
	# Generate links to current git commit
	python gitlinks.py
	if [ "${TECTONIC}" = "true" ]; then\
		tectonic starry.tex --print --keep-logs;\
	else\
		${LATEX} starry.tex;\
		( ${CHECK_RERUN} && ${LATEX} starry.tex ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} starry.tex ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} starry.tex ) || echo "Done.";\
	fi
	# Update the proofs page in the docs
	python genproofs.py

clean:
	$(RM_TMP)
	rm -rf starryNotes.bib
	rm -rf $(FIGURES)

figures/%.pdf: figures/%.py
	cd $(<D);python $(<F)

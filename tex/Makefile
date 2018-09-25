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
#TEXFILE     = "\def\hideedits{}\input{starry}"
TEXFILE     = "\input{starry}"

default: starry.pdf

starry.pdf: starry.tex stylez.tex starry.bib $(FIGURES) $(SOURCES)
	# Generate links to current git commit
	python gitlinks.py
	if [ "${TECTONIC}" = "true" ]; then\
	    echo ${TEXFILE} > tmp.tex;\
		tectonic tmp.tex --print --keep-logs;\
		mv tmp.log starry.log;\
		mv tmp.pdf starry.pdf;\
		rm tmp.tex;\
	else\
		${LATEX} ${TEXFILE};\
		( ${CHECK_RERUN} && ${LATEX} ${TEXFILE} ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} ${TEXFILE} ) || echo "Done.";\
		( ${CHECK_RERUN} && ${LATEX} ${TEXFILE} ) || echo "Done.";\
	fi
	# Update the proofs page in the docs
	python genproofs.py

clean:
	$(RM_TMP)
	rm -rf starryNotes.bib
	rm -rf $(FIGURES)

figures/%.pdf: figures/%.py
	cd $(<D);python $(<F)

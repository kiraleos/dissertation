pandoc -s .\src\dissertation.md -f markdown -t pdf --csl=acm.csl -o .\out\dissertation.pdf --pdf-engine=pdflatex --toc -H ./src/disable_float.tex
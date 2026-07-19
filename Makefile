PYTHON := /Users/ykirpichev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
PDF := output/pdf/principal-ml-systems-handbook.pdf

.PHONY: book verify clean

book:
	$(PYTHON) src/build_book.py --output $(PDF)

verify: book
	$(PYTHON) src/verify_pdf.py $(PDF)

clean:
	rm -f $(PDF)
	rm -f tmp/pdfs/handbook-*.png


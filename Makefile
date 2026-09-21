PYTHON ?= python3
PDF := output/pdf/engineering-large-language-models.pdf
PREVIEW_PAGES ?= 1 2 3 9 100 200

.PHONY: book test verify check-links previews release clean

book:
	$(PYTHON) src/build_book.py --output $(PDF)

test:
	$(PYTHON) -m unittest discover -s tests -v

verify: test book
	$(PYTHON) src/verify_pdf.py $(PDF)

check-links:
	$(PYTHON) src/check_links.py manuscript

previews: book
	$(PYTHON) src/render_previews.py $(PDF) $(PREVIEW_PAGES)

release: verify
	$(PYTHON) -m src.package_release

clean:
	rm -f $(PDF)
	rm -rf tmp/pdfs/review-contact tmp/pdfs/verification
	rm -f output/previews/page-*.png

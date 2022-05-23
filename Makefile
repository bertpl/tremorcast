help:
	@echo 'Commands:'
	@echo '  format					Run isort & black.'
	@echo '  strip_notebooks		Strips output cells from Jupyter notebooks.'
	@echo '  test					Run unit tests.'


format:
	isort .
	black .

strip_notebooks:
	bash strip_notebooks.sh .

test:
	pytest
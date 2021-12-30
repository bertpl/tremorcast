help:
	@echo 'Commands:'
	@echo '  format					Run isort & black.'
	@echo '  test					Run unit tests.'


format:
	isort .
	black .


test:
	pytest
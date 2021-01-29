NOTEBOOK = src/notebook.md
BUILD = out/notebook.md

build: ${BUILD}

${BUILD}: src/notebook.ipynb src/jupyter_nbconvert_config.py
	mkdir -p $(dir $@)
	poetry run jupyter nbconvert \
		--config=src/jupyter_nbconvert_config.py \
		--stdout \
		$< \
		> $@

up:
	poetry run jupyter-notebook ${NOTEBOOK}

bootstrap:
	poetry install

preview: ${BUILD}
	poetry run python -m rich.markdown $<

src/notebook.ipynb: ${NOTEBOOK}
	poetry run jupytext --sync ${NOTEBOOK}

fmt: ${NOTEBOOK}
	poetry run blacken-docs $< || true
	docker-compose run --rm prettier \
		npx prettier \
			--write "/mnt/*.md" \
			--prose-wrap always

clean:
	rm ${BUILD}

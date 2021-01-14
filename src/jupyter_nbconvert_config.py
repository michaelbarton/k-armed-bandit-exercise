import re
import rich
import textwrap
import typing

from nbconvert.preprocessors import base
from rich import traceback
import io
import nbformat
import rich

traceback.install()


c.NbConvertApp.export_format = "markdown"


class RemoveMagicPreprocessor(base.Preprocessor):
    """Remove magic from cell."""

    def preprocess_cell(
        self, cell: nbformat.NotebookNode, resources: typing.Dict[str, typing.Any], index: int
    ) -> typing.Tuple[nbformat.NotebookNode, typing.Dict[str, typing.Any]]:
        """Remove magic from cell."""

        if cell["cell_type"] == "code":
            if "%%run_pytest" in cell["source"]:
                cell["source"] = cell["source"].split("\n", 2)[2].lstrip()

        return cell, resources


class FixMarkdownLinkBreak(base.Preprocessor):
    """Remove line breaks in markdown links."""

    def preprocess_cell(
        self, cell: nbformat.NotebookNode, resources: typing.Dict[str, typing.Any], index: int
    ) -> typing.Tuple[nbformat.NotebookNode, typing.Dict[str, typing.Any]]:
        """Remove broken line breaks in markdown links."""

        if cell["cell_type"] == "markdown":
            cell["source"] = re.sub("(]:\n  http)", "]: http", cell["source"])

        return cell, resources


class AddArticleMetadata(base.Preprocessor):
    """Add article metadata."""

    def preprocess(
        self, notebook: nbformat.NotebookNode, resources: typing.Dict[str, typing.Any]
    ) -> typing.Tuple[nbformat.NotebookNode, typing.Dict[str, typing.Any]]:
        """Add article frontmatter YAML."""
        metadata = dict(notebook["metadata"]["blog"])

        node = nbformat.from_dict(
            {
                "metadata": {},
                "cell_type": "markdown",
                "source": textwrap.dedent(
                    f"""\
                    ---
                    "kind": "article"
                    "date": "{metadata["date"]}"
                    "title": "{metadata["title"]}"
                    ---
                """
                ),
            }
        )

        notebook.cells.insert(0, node)
        return notebook, resources


# Configure tag processor to remove cells based on tags
c.TagRemovePreprocessor.remove_cell_tags = ("remove_cell",)
c.TagRemovePreprocessor.remove_all_outputs_tags = ("remove_output",)


c.RemoveMagicPreprocessor.enabled = True
c.FixMarkdownLinkBreak.enabled = True
c.AddArticleMetadata.enabled = True


c.Exporter.default_preprocessors = [
    "nbconvert.preprocessors.TagRemovePreprocessor",
    RemoveMagicPreprocessor,
    FixMarkdownLinkBreak,
    AddArticleMetadata,
]

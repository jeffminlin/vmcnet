"""Generate the API reference pages and mkdocs nav."""
from pathlib import Path

import mkdocs_gen_files

# flax.linen.Module gives mkdocstrings problems, so we filter out anything
# that explicitly subclasses it, as well as private methods (which is default)
FILTER_OUT_MODULE = """    handler: python
    selection:
        filters:
        - "!^_[^_]"
        - "!Module"
"""

if __name__ == "<run_path>":
    nav = mkdocs_gen_files.Nav()

    # leave out __init__.py files
    package_paths = Path("vmcnet").glob("**/[!__init__]*.py")

    for path in sorted(package_paths):
        module_path = path.relative_to("vmcnet").with_suffix("")
        doc_path = path.relative_to("vmcnet").with_suffix(".md")
        full_doc_path = Path("api", doc_path)  # get path of file to be created

        parts = list(module_path.parts)
        nav[parts] = doc_path  # append to navigation

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            # put in mkdocstrings magic
            ident = ".".join(("vmcnet",) + module_path.parts)
            print("::: " + ident, file=fd)
            if parts[-1] == "core":
                print(FILTER_OUT_MODULE, file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    # make the navigation file
    with mkdocs_gen_files.open("api/api_nav.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())

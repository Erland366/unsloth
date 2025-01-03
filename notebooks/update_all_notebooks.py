import json
import os
import re
import shutil
from datetime import datetime
from glob import glob

general_announcement_content = """To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions).

**[NEW] As of Novemeber 2024, Unsloth now supports vision finetuning!**

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)"""

installation_content = """%%capture
!pip install unsloth
# Also get the latest nightly Unsloth!
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
"""

installation_kaggle_content = """%%capture
# Kaggle is slow - you'll have to wait 5 minutes for it to install.
!pip install pip3-autoremove
!pip-autoremove torch torchvision torchaudio -y
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
!pip install unsloth"""

new_announcement_content_non_vlm = """* We support Llama 3.2 Vision 11B, 90B; Pixtral; Qwen2VL 2B, 7B, 72B; and any Llava variant like Llava NeXT!
* We support 16bit LoRA via `load_in_4bit=False` or 4bit QLoRA. Both are accelerated and use much less memory!
"""

new_announcement_content_vlm = """**We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! You can also select to finetune the attention or the MLP layers!**"""


def copy_folder(source_path, new_name, destination_path=None, replace=False):
    if destination_path is None:
        destination_path = os.path.dirname(source_path)

    new_path = os.path.join(destination_path, new_name)

    try:
        if replace and os.path.exists(new_path):
            shutil.rmtree(new_path)
            print(f"Removed existing folder: '{new_path}'")

        shutil.copytree(source_path, new_path)
        print(f"Successfully copied '{source_path}' to '{new_path}'")
    except FileNotFoundError:
        print(f"Error: Source folder '{source_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def is_path_contains_any(file_path, words):
    return any(re.search(word, file_path, re.IGNORECASE) for word in words)


def update_notebook_sections(
    notebook_path,
    general_announcement,
    installation_steps,
    installation_steps_kaggle,
    new_announcement_non_vlm,
    new_announcement_vlm,
):
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = json.load(f)

        updated = False
        i = 0
        while i < len(notebook_content["cells"]):
            cell = notebook_content["cells"][i]

            if cell["cell_type"] == "markdown":
                source_str = "".join(cell["source"]).strip()

                if source_str == "# General":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
                    ):
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in general_announcement.splitlines()
                        ]
                        updated = True
                        i += 1
                elif source_str == "# News":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "markdown"
                    ):
                        if is_path_contains_any(notebook_path, ["Vision"]):
                            announcement = new_announcement_vlm
                        else:
                            announcement = new_announcement_non_vlm
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in announcement.splitlines()
                        ]
                        updated = True
                        i += 1
                elif source_str == "# Installation":
                    if (
                        i + 1 < len(notebook_content["cells"])
                        and notebook_content["cells"][i + 1]["cell_type"] == "code"
                    ):
                        # Use installation_steps_kaggle if "Kaggle" is in the path,
                        # otherwise use installation_steps
                        if is_path_contains_any(notebook_path, ["kaggle"]):
                            installation = installation_steps_kaggle
                        else:
                            installation = installation_steps
                        notebook_content["cells"][i + 1]["source"] = [
                            f"{line}\n" for line in installation.splitlines()
                        ]
                        updated = True
                        i += 1

            i += 1

        # Ensure GPU metadata is set
        if "metadata" not in notebook_content:
            notebook_content["metadata"] = {}
        if "accelerator" not in notebook_content["metadata"]:
            notebook_content["metadata"]["accelerator"] = "GPU"
            updated = True
        if "colab" not in notebook_content["metadata"]:
            notebook_content["metadata"]["colab"] = {"provenance": []}
            updated = True
        if "kernelspec" not in notebook_content["metadata"]:
            notebook_content["metadata"]["kernelspec"] = {
                "display_name": "Python 3",
                "name": "python3",
            }
            updated = True

        if updated:
            with open(notebook_path, "w", encoding="utf-8") as f:
                json.dump(notebook_content, f, indent=1)
            print(f"Updated: {notebook_path}")
        else:
            print(f"No sections found to update in: {notebook_path}")

    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in notebook at {notebook_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {notebook_path}: {e}")


def main():
    notebook_directory = "notebooks"
    notebook_pattern = "*.ipynb"

    notebook_files = glob(os.path.join(notebook_directory, notebook_pattern))

    if not notebook_files:
        print(
            f"No notebooks found in the directory: {notebook_directory} with pattern: {notebook_pattern}"
        )
        return

    for notebook_file in notebook_files:
        update_notebook_sections(
            notebook_file,
            general_announcement_content,
            installation_content,
            installation_kaggle_content,
            new_announcement_content_non_vlm,
            new_announcement_content_vlm,
        )


def update_readme(readme_path, notebooks_dir, type_order=None):
    base_url_colab = (
        "https://colab.research.google.com/github/Erland366/notebooks/blob/master/"
    )
    base_url_kaggle = "https://www.kaggle.com/notebooks/welcome?src=https://github.com/Erland366/notebooks/blob/master/"

    paths = glob(os.path.join(notebooks_dir, "*.ipynb"))

    sections = {
        "LLM": {"header": "## LLM Notebooks\n", "subsections": {}},
        "Vision": {"header": "## Vision Notebooks\n", "rows": ""},
    }

    table_header = (
        "| Model | Type | Colab Link | Kaggle Link |\n| --- | --- | --- | --- |\n"
    )

    for path in paths:
        notebook_name = os.path.basename(path)
        kaggle_path = path.replace("notebooks/", "notebooks/Kaggle-")
        model = ""
        type_ = ""
        colab_link = ""
        kaggle_link = ""

        if is_path_contains_any(path, ["kaggle"]):
            continue
            # section_name = "Kaggle"
            # kaggle_link = (
            #     f"[Open in Kaggle]({base_url_kaggle}{path}?accelerator=nvidiaTeslaT4)"
            # )

            # parts = notebook_name.replace(".ipynb", "").split("-")
            # # Special handling for Kaggle names
            # if len(parts) >= 2 and parts[0].lower() == "kaggle":
            #     model = parts[1].replace("_", " ")
            #     type_ = parts[2].replace("_", " ") if len(parts) >= 3 else ""
            # else:
            #     model = parts[0].replace("_", " ")
            #     type_ = parts[-1].replace("_", " ")

            # sections[section_name]["rows"] += (
            #     f"| {model} | {type_} | {colab_link} | {kaggle_link}\n"
            # )

        elif is_path_contains_any(path, ["vision"]):
            section_name = "Vision"
            colab_link = f"[Open in Colab]({base_url_colab}{path})"
            kaggle_link = f"[Open in Kaggle]({base_url_kaggle}{kaggle_path}&accelerator=nvidiaTeslaT4)"
            parts = notebook_name.replace(".ipynb", "").split("-")
            model = parts[0].replace("_", " ")
            type_ = parts[-1].replace("_", " ")

            sections[section_name]["rows"] += (
                f"| {model} | {type_} | {colab_link} | {kaggle_link}\n"
            )

        else:
            section_name = "LLM"
            colab_link = f"[Open in Colab]({base_url_colab}{path})"
            kaggle_link = f"[Open in Kaggle]({base_url_kaggle}{kaggle_path}&accelerator=nvidiaTeslaT4)"
            parts = notebook_name.replace(".ipynb", "").split("-")
            model = parts[0].replace("_", " ")
            type_ = parts[-1].replace("_", " ")

            if type_ not in sections[section_name]["subsections"]:
                sections[section_name]["subsections"][type_] = ""
            sections[section_name]["subsections"][type_] += (
                f"| {model} | {type_} | {colab_link} | {kaggle_link}\n"
            )

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        start_marker = "# ✨ Fine-tuning Notebooks"
        start_index = readme_content.find(start_marker)
        if start_index == -1:
            raise ValueError(
                "Start marker '# ✨ Fine-tuning Notebooks' not found in README."
            )
        start_index += len(start_marker)

        end_marker = "<!-- End of Notebook Links -->"
        end_index = readme_content.find(end_marker)
        if end_index == -1:
            raise ValueError(
                "End marker '<!-- End of Notebook Links -->' not found in README."
            )

        content_before = readme_content[:start_index]
        content_after = readme_content[end_index:]

        updated_notebooks_links = ""

        if sections["LLM"]["subsections"]:
            updated_notebooks_links += sections["LLM"]["header"] + table_header

            if type_order:
                sorted_types = sorted(
                    sections["LLM"]["subsections"].keys(),
                    key=lambda x: type_order.index(x)
                    if x in type_order
                    else float("inf"),
                )
            else:
                sorted_types = sorted(sections["LLM"]["subsections"].keys())

            for type_ in sorted_types:
                rows = sections["LLM"]["subsections"][type_]
                updated_notebooks_links += rows

        for section_name in ["Vision"]:
            if sections[section_name]["rows"]:
                updated_notebooks_links += (
                    sections[section_name]["header"]
                    + table_header
                    + sections[section_name]["rows"]
                )

        timestamp = f"<!-- Last updated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n"

        updated_readme_content = (
            content_before + "\n" + updated_notebooks_links + timestamp + content_after
        )

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated_readme_content)

        print(f"Successfully updated {readme_path}")

    except FileNotFoundError:
        print(f"Error: {readme_path} not found.")
    except Exception as e:
        raise e
        # print(f"An error occurred while updating {readme_path}: {e}")


def copy_and_update_notebooks(
    template_dir,
    destination_dir,
    general_announcement,
    installation,
    installation_kaggle,
    new_announcement_non_vlm,
    new_announcement_vlm,
):
    """Copies notebooks from template_dir to destination_dir, updates them, and renames them."""
    template_notebooks = glob(os.path.join(template_dir, "*.ipynb"))

    for template_notebook_path in template_notebooks:
        notebook_name = os.path.basename(template_notebook_path)

        colab_notebook_name = notebook_name
        destination_notebook_path = os.path.join(destination_dir, colab_notebook_name)

        shutil.copy2(template_notebook_path, destination_notebook_path)
        print(f"Copied '{colab_notebook_name}' to '{destination_dir}'")

        update_notebook_sections(
            destination_notebook_path,
            general_announcement,
            installation,
            installation_kaggle,
            new_announcement_non_vlm,
            new_announcement_vlm,
        )

        kaggle_notebook_name = "Kaggle-" + notebook_name
        destination_notebook_path = os.path.join(destination_dir, kaggle_notebook_name)

        shutil.copy2(template_notebook_path, destination_notebook_path)
        print(f"Copied '{kaggle_notebook_name}' to '{destination_dir}'")

        update_notebook_sections(
            destination_notebook_path,
            general_announcement,
            installation_kaggle,
            installation_kaggle,
            new_announcement_non_vlm,
            new_announcement_vlm,
        )


if __name__ == "__main__":
    copy_and_update_notebooks(
        "original_template",
        "notebooks",
        general_announcement_content,
        installation_content,
        installation_kaggle_content,
        new_announcement_content_non_vlm,
        new_announcement_content_vlm,
    )
    main()

    notebook_directory = "notebooks"
    readme_path = "README.md"
    type_order = [
        "Alpaca",
        "Conversational",
        "CPT",
        "DPO",
        "ORPO",
        "Text_Completion",
        "CSV",
        "Inference",
        "Unsloth_Studio",
    ]  # Define your desired order here
    update_readme(readme_path, notebook_directory, type_order)

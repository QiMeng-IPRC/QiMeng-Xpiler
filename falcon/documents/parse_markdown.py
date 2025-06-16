import json

import markdown
from bs4 import BeautifulSoup


def parse_markdown(file_path):
    """解析 Markdown 文件并提取标题及其对应的段落内容."""
    # Read the Markdown file
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert Markdown to HTML.
    html_content = markdown.markdown(md_content)

    # Use BeautifulSoup to parse HTML.
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the title and paragraphs.
    titles_and_paragraphs = []
    current_title = None

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
        if element.name.startswith("h"):  # Determine if it is a title.
            # If the current title is not empty, save the previous title and
            # paragraph.
            if current_title is not None:
                titles_and_paragraphs.append((current_title, paragraphs))
            current_title = element.get_text()  # Update the current title.
            paragraphs = []  # Reset paragraph list
        elif element.name == "p" and current_title is not None:
            paragraphs.append(
                element.get_text()
            )  # Add a paragraph to the current section's list of paragraphs.

    # Add the final title and paragraph.
    if current_title is not None:
        titles_and_paragraphs.append((current_title, paragraphs))

    return titles_and_paragraphs


if __name__ == "__main__":
    # Enter the Markdown file path.
    # Replace with your Markdown file path.
    markdown_file = "cntoolkit_3.5.2_cambricon_bang_c_4.5.1.md"

    # Analyze Markdown documents
    titles_and_paragraphs = parse_markdown(markdown_file)
    BANG_DOC = {}
    # Print the extracted titles and corresponding paragraphs.
    for title, paragraphs in titles_and_paragraphs:
        if "bang" in title or "memcpy" in title:
            inst_name = title.split(" ")[1]
            print(f"Title: {paragraphs}")
            BANG_DOC[inst_name] = " ".join(paragraphs)

    with open("./bang_c_user_guide.json", "w", encoding="utf8") as json_file:
        json.dump(BANG_DOC, json_file, ensure_ascii=False, indent=2)

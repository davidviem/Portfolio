import os
from bs4 import BeautifulSoup

# Function to merge HTML files and wrap plots in containers
def merge_html_files(input_dir, output_file):
    # Initialize the final HTML content
    final_html = '<div style="display: flex; flex-direction: column; align-items: center;">'

    # Iterate through each file in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".html"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                # Read the content of the HTML file
                html_content = file.read()

                # Wrap the HTML content in a container
                wrapped_content = f'<div style="width: 70%; margin-bottom: 20px;">{html_content}</div>'

                # Append the wrapped content to the final HTML
                final_html += wrapped_content

    final_html += '</div>'

    # Parse the final HTML using BeautifulSoup
    soup = BeautifulSoup(final_html, "html.parser")

    # Write the merged HTML content to the output file
    with open(output_file, "w", encoding="utf-8") as output:
        output.write(str(soup))

# Directory containing HTML files
input_directory = r"./All Point Intensity Distribution"

# Output file name
output_file = "All Point Intensity Distribution.html"

# Call the function to merge HTML files
merge_html_files(input_directory, output_file)

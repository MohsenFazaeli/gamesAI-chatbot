from shared.constants.constants import *

# Specify the path to your HTML
html_file_path = synthetic_map_html




# Read the existing content of the HTML file
with open(html_file_path, 'r') as file:
    html_content = file.read()

with open(head_tail_path, 'r') as file:
    head_content = file.read()

with open(body_tail_path, 'r') as file:
    body_content = file.read()

insert_position = html_content.find('</head>')
# Insert the custom elements string at the end of the body
modified_html_content = html_content[:insert_position] + head_content + html_content[insert_position:]
print(insert_position)


insert_position = modified_html_content.rfind('</body>')
print(insert_position)
# Insert the custom elements string at the end of the body
modified_html_content = modified_html_content[:insert_position] + body_content + modified_html_content[insert_position:]

modified_html_content = modified_html_content.replace('<div class="folium-map', '<div class="folium-map some-div-behind', 1)


# Write the modified content back to the HTML file
with open(index_html_path, 'w') as file:
    file.write(modified_html_content)
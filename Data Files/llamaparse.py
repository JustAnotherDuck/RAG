from llama_parse import LlamaParse

import re


LLAMA_CLOUD_API_KEY='llx-4Wuo5LjJuZCsNkyzVvsb73wzkshHPEEufzr8airaJ9hXFfel'

import nest_asyncio
nest_asyncio.apply()  # Required for async in notebooks

# Initialize parser
parser = LlamaParse(
    api_key="llx-4Wuo5LjJuZCsNkyzVvsb73wzkshHPEEufzr8airaJ9hXFfel",  # Or set via environment variable
    result_type="text",  # Options: "markdown", "text"
    verbose=True,
    language="en"  # Optional, default is English
)

# Parse a single file
file_path = "files/2407.13035v1.pdf"
documents = parser.load_data(file_path)
extracted = re.search(r'/([^/]+)\.pdf$', file_path).group(1)

# Access parsed content
file = open(f"{extracted}_llamaparse.txt", "w") 
for doc in documents:
    file.write(doc.text)  # Writes markdown content to Employees.txt

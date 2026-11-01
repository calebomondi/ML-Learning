# line_count.py
import sys

count = 0
for line in sys.stdin:
    count += 1

# print goes to sys.stdout
print(count)

# step 2: chain them, filter then count
# cat getting_data/file.txt | uv run getting_data/regex.py "[0-9]" | uv run getting_data/line_count.py
# 4
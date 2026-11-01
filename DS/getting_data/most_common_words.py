import sys
from collections import Counter

try:
    # pass in number of words as first argument
    num_words = int(sys.argv[1])
except:
    print("usage: most_common_words.py num_words")
    sys.exit(1)

counter = Counter(
    word.lower()                        # lowercase words
    for line in sys.stdin
    for word in line.strip().split()    # split on spaces
    if word                             # skip empty 'words'
)

for word, count in counter.most_common(num_words):
    sys.stdout.write(str(count))
    sys.stdout.write("\t")
    sys.stdout.write(word)
    sys.stdout.write("\n")

# cat getting_data/words.txt | uv run getting_data/most_common_words.py 5

# 9       data
# 6       the
# 4       science
# 4       is
# 4       python

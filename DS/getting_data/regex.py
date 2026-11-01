# egrep.py
import sys, re

# sys.argv is the list of the command-line arguments
# sys.argv[0] is the name of the program itself
# sys.argv[1] will be the regex specified at the command line
regex = sys.argv[1]

# for every line passed into the script
for line in sys.stdin:
    # if it matches the regex, write it to stdout
    if re.search(regex, line):
        sys.stdout.write(line)


# step 1: show only the lines containing a digit
# cat getting_data/file.txt | uv run getting_data/regex.py "[0-9]"
# I have 3 cats and 2 dogs
# hey there, the meeting is at 10am
# python 3.12 is installed
# call me at 555-1234
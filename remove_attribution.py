input_file = 'f:\\Download\\fra-eng\\eng-fra.txt'
output_file = 'f:\\Download\\fra-eng\\eng-fra_cleaned.txt'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Split the line by tab and remove the last part (attribution)
        parts = line.split('\t')
        cleaned_line = '\t'.join(parts[:2])  # Keep only the first two parts
        outfile.write(cleaned_line + '\n')

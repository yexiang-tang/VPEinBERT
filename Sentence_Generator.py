# This program generates sentences with CFG. All sentences are stored in a CSV file

import nltk
import ssl
import stanza
from nltk.parse.generate import generate, demo_grammar
from nltk.draw.tree import draw_trees
from nltk import CFG
import csv


# The setence generator works by specifying sentence structures. 
# Adjust, comment, or uncomment pasrt accordingly to your need
vpe = CFG.fromstring("""
S -> Phrase_P 'and' Omit_P
S -> Phrase_P 'but' Omit_N
S -> Phrase_N 'but' Omit_P
S -> Phrase_PA 'and' Omit_PA
S -> Phrase_PA 'but' Omit_NA
S -> Phrase_NA 'but' Omit_PA
S -> Phrase_PP 'and' Omit_PP
S -> Phrase_PP 'but' Omit_NP
S -> Phrase_NP 'but' Omit_PP
Phrase_P -> NPS VP_P
VP_P -> V_SG N
VP_N -> 'does' Neg V
Omit_P -> NPS 'does'
Omit_N -> NPS 'does' Neg
Phrase_N -> NPS 'does' Neg VP
Phrase_PA -> NPS Aux VP
Omit_PA -> NPS Aux
Omit_NA -> NPS Aux_N
Phrase_NA -> NPS Aux_N VP
Phrase_PP -> NPS VP_Pa
Omit_PP -> NPS 'did'
Omit_NP -> NPS "didn't"
Phrase_NP -> NPS 'did not' VP

NPS -> 'Tim' | 'Sarah' | 'Mary' | 'John'
Neg -> 'not'
Aux -> 'can' | 'will'
Aux_N -> "cannot" | "won't"

VP -> V N
VP_Pa -> V_Pa N

V -> 'like' | 'play' | 'read' | 'watch' |'study' | 'teach' |'cook'
V_SG -> 'likes' | 'plays' | 'reads' | 'watches' |'studies' | 'teaches' |'cooks'
V_Pa -> 'liked' | 'played' | 'read' | 'watched' |'studied' | 'taught' |'cooked'
N -> 'coffee' | 'tea' | 'chess' | 'music'|| 'novel' | 'tennis' | 'math' | 'history' | 'pizza' | 'sushi' | 'movies' | 'TV shows' | 'poems'
""")



# Generate sentences
sentences_vpe = list(generate(vpe, depth=10))

tokenized_sentences_vpe = [' '.join(sentence).split() for sentence in sentences_vpe]

print("Generated Sentences:")
for sentence in sentences_vpe:
    print(' '.join(sentence))  # Display generated sentences

# Parse each sentence and collect unique trees
parser = nltk.ChartParser(vpe)

# Store sentences to CSV
output_csv = r"C:\Users\Tang Yexiang\Desktop\Thesis_code\Generated sentences\VPE_original.CSV"
string_sentences = [' '.join(tokens) for tokens in sentences_vpe]

with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["VPE"])  # Add a header
    for sentence in string_sentences:
        writer.writerow([sentence])  # Write each sentence as a row

print(f"Sentences have been saved to {output_csv}")

# Draw trees, comment or uncomment based on need
parses = []
for sentence in sentences_vpe:
    for tree in parser.parse(sentence):
        parses.append(tree)
        print(tree)
draw_trees(*parses)

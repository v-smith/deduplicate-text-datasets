from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import spacy

nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

doc = nlp("pt admitted to nicu")

# Let's look at a random entity!
entity = doc.ents[1]

print("Name: ", entity)

linker = nlp.get_pipe("scispacy_linker")
for umls_ent in entity._.kb_ents:
	print(linker.kb.cui_to_entity[umls_ent[0]])

a=1

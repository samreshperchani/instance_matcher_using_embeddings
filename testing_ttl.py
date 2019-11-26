import rdflib

catgeory_labels_file = open('category_labels.ttl', 'w+', encoding='utf-8')
instance_labels_file = open('instance_labels.ttl', 'w+', encoding='utf-8')
property_labels_file = open('property_labels.ttl', 'w+', encoding='utf-8')
class_labels_file = open('class_labels.ttl', 'w+', encoding='utf-8')
abstract_file = open('abstracts.ttl', 'w+', encoding='utf-8')

g=rdflib.Graph()
g.load('download.xml')

for s,p,o in g:
    if p.lower().strip() == 'http://www.w3.org/2000/01/rdf-schema#label':
        if s.lower().find('/resource/category:') != -1:
            catgeory_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
        elif s.lower().find('/resource/') != -1:
            instance_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
        elif s.lower().find('/property/') != -1:
            property_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
        elif s.lower().find('/class/') != -1:
            class_labels_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
    
    elif p.lower().strip() == 'http://dbkwik.webdatacommons.org/ontology/abstract':
        abstract_file.write('<' + s.strip() + '> <' +  p.strip() +  '> \"' + o.strip() +  '\" .' + '\n')
        g.remove( (s, p, o) )

g.serialize(destination='download_revised.ttl', format='nt')

#for s,p,o
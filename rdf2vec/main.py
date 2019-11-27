from rdf2vec import RDF2Vec


if __name__ == '__main__':
    
    # create object of a class
    rdf2vec_model = RDF2Vec()

    
    #insert labels to db
    rdf2vec_model.process_labels()
    
    # revise uri based on duplicate labels
    rdf2vec_model.revise_uris()

    # generate labels mapping files
    rdf2vec_model.generate_labels_mapping_file()
    
    # extract text
    rdf2vec_model.generate_knowledge_graphs()

    # revised knowledge graphs with labels mapping
    rdf2vec_model.generate_revised_knowledge_graphs()

    # generate final knowledge graph for random walks
    rdf2vec_model.generate_merged_kg_graphs()

    # generate walks on a graph
    rdf2vec_model.generate_walks()

    # train RDF2Vec model
    rdf2vec_model.train_model()

    # extract vectors from rdf2vec model
    #rdf2vec_model.extract_vectors('darkscape','oldschoolrunscape')
    
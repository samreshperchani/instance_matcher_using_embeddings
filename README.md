# instance_matcher_using_embeddings
The project uses three types of embeddings (DOC2Vec, Word2Vec, RDF2Vec) to perform instance matching on Knowledge Graphs

# Setup configuration parameters
Please set appropriate paraemters including path and database information in config.py file inside config folder.
Database is required to group same label entities in case of RDF2Vec model


# General Steps
1. Setup Database with Appropriate Name
2. Execute create_tables.sql script ptovided in db_scripts\ddl_scripts folder
3. Create DSN connection to DB
4. Provide DSN Name in "LABELS DATABASE CONFIGURATION (please input DSN name)" section of config file
5. The configuration lets you to set different databases for instance, category, property and class labels [It is useful if DB size is limited]

# Steps to execute DBkWik workflow
1. Create directory with name "data"
2. Create "data_dumps" inside "data" directory [Please note that the each dump should be .tar.gz file containing all ttl files and media wiki xml file of the respective wiki]
3. Create directory with name "gold_standard" in "data" directory
4. Place gold standard files in OAEI xml format inside "gold_standard" directory.


# Steps to execute OAEI Matcher
1. Create directory with name "data"
2. Create subdirectory inside "data" namely "xml_dumps"
3. Copy XML dumps in "data/xml_dumps" directory"
4. Execute main_oaei.py script

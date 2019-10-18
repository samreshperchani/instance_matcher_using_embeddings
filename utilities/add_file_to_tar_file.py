import tarfile

TAR_FILE_PATH = 'E:/University/Thesis/instance_matcher_using_embeddings/data/data_dumps/323~en~memory-beta.tar.gz'
XML_FILE_PATH = 'E:/University/Thesis/instance_matcher_using_embeddings/data/data_dumps/memory_betawikiacom-20181006-current.xml'


tar = tarfile.open(TAR_FILE_PATH, mode='a:')

tar.add(XML_FILE_PATH)

tar.close()
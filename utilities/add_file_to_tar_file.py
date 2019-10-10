import tarfile

TAR_FILE_PATH = 'E:/University/Thesis/instance_matcher_using_embeddings/data/data_dumps/130814~en~gameofthrones~gameofthrones.wikia.com.tar.gz'
XML_FILE_PATH = 'E:/University/Thesis/instance_matcher_using_embeddings/data/data_dumps/gameofthroneswikiacom-20181008-current.xml'


tar = tarfile.open(TAR_FILE_PATH, mode='a:')

tar.add(XML_FILE_PATH)

tar.close()
import re
from xml.etree.cElementTree import iterparse  # LXML isn't faster, so let's go with the built-in solution


def get_namespace(tag):
    """Get the namespace of tag.
    Parameters
    ----------
    tag : str
        Namespace or tag.
    Returns
    -------
    str
        Matched namespace or tag.
    """
    m = re.match("^{(.*?)}", tag)
    namespace = m.group(1) if m else ""
    if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
        raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
    return namespace


def extract_pages(f, wiki_name, filter_namespaces):
    """Extract pages from a MediaWiki database dump.
    Parameters
    ----------
    f : file
        File-like object.
    filter_namespaces : set of str
         Namespaces that will be extracted.
    Yields
    ------
    tuple of (str or None, str, str)
        Title, text and page id.
    """
    elems = (elem for _, elem in iterparse(f, events=("end",)))

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    elem = next(elems)
    namespace = get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    ns_path = "./{%(ns)s}ns" % ns_mapping
    pageid_path = "./{%(ns)s}id" % ns_mapping

    for elem in elems:
        if elem.tag == page_tag:
            ns = elem.find(ns_path).text
            if ns in filter_namespaces:
                title = elem.find(title_path).text
                pageid = elem.find(pageid_path).text
                text = elem.find(text_path).text
                yield title, text, pageid, wiki_name

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # except that we don't need to prune backlinks from the parent
            # because we don't use LXML.
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. The pages comprise the bulk of the
            # file, so in practice we prune away enough.
            elem.clear()

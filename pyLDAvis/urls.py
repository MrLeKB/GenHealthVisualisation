"""
LDAvis URLs
==========
URLs and filepaths for the LDAvis javascript libraries
"""

import os
from . import __path__, __version__

__all__ = ["D3_URL", "LDAVIS_URL", "LDAVIS_CSS_URL",
           "D3_LOCAL", "LDAVIS_LOCAL", "LDAVIS_CSS_LOCAL"]

D3_URL = "https://d3js.org/d3.v5.js"

#DEV = 'git' in __version__
LOCAL_JS_DIR = os.path.join(__path__[0], "js")
D3_LOCAL = os.path.join(LOCAL_JS_DIR, "d3.v5.min.js")

# Avoid browser caching with @version in the URL.
#WWW_JS_DIR = "https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@{0}/pyLDAvis/js/".format(__version__)

#WWW_JS_DIR = "https://mrlekb.github.io/js/".format(__version__)


LDAVIS_URL = "/pyLDAvis/js/ldavis.js"
LDAVIS_CSS_URL = "/pyLDAvis/js/ldavis.css"

LDAVIS_LOCAL = "/pyLDAvis/js/ldavis.js"
LDAVIS_CSS_LOCAL = "/pyLDAvis/js/ldavis.css"

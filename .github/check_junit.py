"""Fail unless a pytest JUnit report shows tests that ran without failures or errors.

Used by the tests workflow instead of pytest's exit code, which a crash of
torch at interpreter exit can turn nonzero after a successful run.
"""

import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors")}
print("tests: {tests}, failures: {failures}, errors: {errors}".format(**counts))
sys.exit(0 if counts["tests"] > 0 and counts["failures"] + counts["errors"] == 0 else 1)

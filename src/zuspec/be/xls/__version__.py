# BASE is the released version and is what a `v<BASE>` tag must match; CI
# rejects a tag that disagrees with it. SUFFIX is stamped by CI for
# non-release builds so a development wheel can never collide with a release.
# 0.1.0 to match the rest of the zuspec estate (see zuspec-ir-core).
BASE="0.1.0"
SUFFIX=""
VERSION="%s%s" % (BASE, SUFFIX)
version=VERSION

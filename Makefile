all: README README.org

# a multiple-target pattern rule means that a single invocation of the command
# builds all the targets, which is what I want here
%EADME %EADME.org: numpysane.py README.footer.org extract_README.py
	python3 extract_README.py numpysane

test:  test2  test3
check: check2 check3
check2: test2
check3: test3
test2 test3:
	python$(patsubst test%,%,$@) test_numpysane.py
.PHONY: check check2 check3 test test2 test3

# make distribution tarball
dist:
	python3 setup.py sdist
.PHONY: dist

# make and upload the distribution tarball
dist_upload:
	python3 setup.py sdist upload
.PHONY: dist_upload

EXTRA_CLEAN += README.org README






########## for testing the python wrapping

PYTHON_VERSION_FOR_EXTENSIONS := 3
include Makefile.common.header

PROJECT_NAME := numpysane
VERSION      := POISON

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
pywrap-sample/testlib_pywrap_GENERATED.o: CFLAGS += -Wno-cast-function-type -Wno-missing-field-initializers

pywrap-sample/testlib_pywrap_GENERATED.o: CFLAGS += $(PY_MRBUILD_CFLAGS)

pywrap-sample/testlibmodule$(PY_EXT_SUFFIX): pywrap-sample/testlib_pywrap_GENERATED.o pywrap-sample/testlib.o
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@

DIST_PY3_MODULES := pywrap-sample/testlibmodule

all: pywrap-sample/testlibmodule$(PY_EXT_SUFFIX)
EXTRA_CLEAN += pywrap-sample/testlibmodule*.so pywrap-sample/*.o pywrap-sample/*.d

include Makefile.common.footer

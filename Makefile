########## For the test suite. To build the extension module for testing the
########## broadcasting-in-C
# Minimal part of https://github.com/dkogan/mrbuild to provide python Makefile
# rules
#
# I support both python2 and python3, but only one at a time
PYTHON_VERSION_FOR_EXTENSIONS := 3
include Makefile.common.header

# I build a python extension module called "testlib" from the C library
# (testlib) and from the numpysane_pywrap wrapper. The wrapper is generated with
# genpywrap.py
test/testlib$(PY_EXT_SUFFIX): test/testlib_pywrap_GENERATED.o test/testlib.o
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@
test/testlib_pywrap_GENERATED.o: CFLAGS += $(PY_MRBUILD_CFLAGS)

CC ?= gcc
CFLAGS += -g
test/testlib.o: test/testlib.c
	$(CC) -Wall -Wextra $(CFLAGS) $(CPPFLAGS) -fPIC -c -o $@ $<

test/testlib_pywrap_GENERATED.c: test/genpywrap.py numpysane_pywrap.py $(wildcard pywrap-templates/*.c)
	./$< > $@

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
test/testlib_pywrap_GENERATED.o: CFLAGS += -Wno-cast-function-type
test/testlib_pywrap_GENERATED.o: test/testlib.h

CFLAGS += -Wno-missing-field-initializers

clean:
	rm -rf test/*.[do] test/*.o test/*.so test/*.so.* test/testlib_pywrap_GENERATED.c README.org README
.PHONY: clean


####### Everything non-extension-module related
.DEFAULT_GOAL := all
all: README README.org README-pywrap README-pywrap.org

# a multiple-target pattern rule means that a single invocation of the command
# builds all the targets, which is what I want here
%EADME %EADME.org: numpysane.py README.footer.org extract_README.py
	python3 extract_README.py numpysane README.org README README.footer.org
%EADME-pywrap %EADME-pywrap.org: numpysane_pywrap.py README.footer.org extract_README.py
	python3 extract_README.py numpysane_pywrap README-pywrap.org README-pywrap README.footer.org

test:  test3
check: check3
check2: test2
check3: test3
test2 test3: test/test-numpysane.py test-c-broadcasting
	python$(patsubst test%,%,$@) test/test-numpysane.py
test-c-broadcasting: test/testlib$(PY_EXT_SUFFIX)
	python${PYTHON_VERSION_FOR_EXTENSIONS} test/test-c-broadcasting.py

.PHONY: check check2 check3 test test2 test3 test-c-broadcasting

DIST_VERSION := $(or $(shell < numpysane.py perl -ne "if(/__version__ = '(.*)'/) { print \$$1; exit}"), $(error "Couldn't parse the distribution version"))

DIST := dist/numpysane-$(DIST_VERSION).tar.gz
$(DIST): README

# make distribution tarball
$(DIST):
	python3 setup.py sdist
.PHONY: $(DIST) # rebuild it unconditionally

dist: $(DIST)
.PHONY: dist

# make and upload the distribution tarball
dist_upload: $(DIST)
	twine upload --verbose $(DIST)
.PHONY: dist_upload



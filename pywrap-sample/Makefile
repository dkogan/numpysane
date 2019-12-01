PYTHON_VERSION_FOR_EXTENSIONS := 3

# Minimal part of https://github.com/dkogan/mrbuild to provide python Makefile
# rules
include Makefile.common.header


all: testlibmodule$(PY_EXT_SUFFIX)
testlibmodule$(PY_EXT_SUFFIX): testlib_pywrap_GENERATED.o testlib.o
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@
testlib_pywrap_GENERATED.o: CFLAGS += $(PY_MRBUILD_CFLAGS)

CC ?= gcc
%.o:%.c
	$(CC) -Wall -Wextra $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

testlib_pywrap_GENERATED.c: genpywrap.py
	./$< > $@


# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
testlib_pywrap_GENERATED.o: CFLAGS += -Wno-cast-function-type

CFLAGS += -Wno-missing-field-initializers

clean:
	rm -rf *.[do] *.so *.so.* testlib_pywrap_GENERATED.c
.PHONY: clean

# keep intermediate files
.SECONDARY:

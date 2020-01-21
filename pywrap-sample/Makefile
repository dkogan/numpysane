PYTHON_VERSION_FOR_EXTENSIONS := 3

# Minimal part of https://github.com/dkogan/mrbuild to provide python Makefile
# rules
include Makefile.common.header

# I build a python extension module called "samplelibmodule" from the C library
# (samplelib) and from the numpysane_pywrap wrapper. The wrapper is generated with
# a demo script genpywrap.py
all: samplelibmodule$(PY_EXT_SUFFIX)
samplelibmodule$(PY_EXT_SUFFIX): samplelib_pywrap_GENERATED.o samplelib.o
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@
samplelib_pywrap_GENERATED.o: CFLAGS += $(PY_MRBUILD_CFLAGS)

CC ?= gcc
%.o:%.c
	$(CC) -Wall -Wextra $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

samplelib_pywrap_GENERATED.c: genpywrap.py
	./$< > $@


# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
samplelib_pywrap_GENERATED.o: CFLAGS += -Wno-cast-function-type

CFLAGS += -Wno-missing-field-initializers

clean:
	rm -rf *.[do] *.so *.so.* samplelib_pywrap_GENERATED.c
.PHONY: clean

# keep intermediate files
.SECONDARY:

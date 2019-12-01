PYTHON_VERSION_FOR_EXTENSIONS := 3
include Makefile.common.header

PROJECT_NAME := numpysane
VERSION      := POISON

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
testlib_pywrap_GENERATED.o: CFLAGS += -Wno-cast-function-type -Wno-missing-field-initializers

testlib_pywrap_GENERATED.o: CFLAGS += $(PY_MRBUILD_CFLAGS)

testlibmodule$(PY_EXT_SUFFIX): testlib_pywrap_GENERATED.o testlib.o
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $^ -o $@

DIST_PY3_MODULES := testlibmodule

all: testlibmodule$(PY_EXT_SUFFIX)
EXTRA_CLEAN += testlibmodule*.so *.o *.d

include Makefile.common.footer

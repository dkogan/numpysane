#define PYMETHODDEF_ENTRY(name,docstring)       \
    { #name,                                    \
      (PyCFunction)__pywrap__ ## name,          \
      METH_VARARGS | METH_KEYWORDS,             \
      docstring },

static PyMethodDef methods[] =
    { FUNCTIONS(PYMETHODDEF_ENTRY)
      {}
    };

#if PY_MAJOR_VERSION == 2

PyMODINIT_FUNC init{MODULE_NAME}(void)
{
    Py_InitModule3("{MODULE_NAME}", methods, "{MODULE_DOCSTRING}");
    import_array();
}

#else

static struct PyModuleDef module_def =
    {
     PyModuleDef_HEAD_INIT,
     "{MODULE_NAME}", "{MODULE_DOCSTRING}",
     -1,
     methods
    };

PyMODINIT_FUNC PyInit_{MODULE_NAME}(void)
{
    PyObject* module = PyModule_Create(&module_def);
    import_array();
    return module;
}

#endif

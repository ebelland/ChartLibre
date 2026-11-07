/* ChartLibre.app's native starter: ChartLibre.py run inside a process whose main executable is built with the
 * current macOS SDK.
 *
 * AppKit gives an app the current look - Liquid Glass window buttons, menus and bars - only when its main
 * executable was linked against a recent SDK; python3 from python-build-standalone says SDK 15, so ChartLibre run
 * by it got the compatibility look. This program loads .python's libpython with dlopen (no version linked in: a
 * newer runtime works too) and calls Py_BytesMain in its own process. When there is no libpython it execs
 * .python/bin/python3, as the app's script did.
 *
 * Built by dev/tools/make_launcher.py (clang, the Command Line Tools) into
 * ChartLibre.app/Contents/MacOS/ChartLibreNative; the script beside it starts it when it is there.
 */
#include <dlfcn.h>
#include <glob.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int (*py_main_t)(int, char **);

/* *path* without its last *levels* components. */
static void up(char *path, int levels) {
    while (levels-- > 0) {
        char *slash = strrchr(path, '/');
        if (slash == NULL) return;
        *slash = '\0';
    }
}

int main(int argc, char **argv) {
    char exe[PATH_MAX], root[PATH_MAX], runtime[PATH_MAX], python[PATH_MAX], script[PATH_MAX], pattern[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, root) == NULL) return 1;
    up(root, 4); /* ChartLibre/ChartLibre.app/Contents/MacOS/ChartLibreNative -> ChartLibre */
    snprintf(runtime, sizeof runtime, "%s/.python", root);
    snprintf(python, sizeof python, "%s/bin/python3", runtime);
    snprintf(script, sizeof script, "%s/ChartLibre.py", root);

    char **args = calloc((size_t)argc + 2, sizeof *args);
    if (args == NULL) return 1;
    args[0] = argv[0];
    args[1] = script;
    for (int index = 1; index < argc; ++index) args[index + 1] = argv[index];

    snprintf(pattern, sizeof pattern, "%s/lib/libpython3.*.dylib", runtime);
    glob_t found;
    if (glob(pattern, 0, NULL, &found) == 0 && found.gl_pathc > 0) {
        void *library = dlopen(found.gl_pathv[found.gl_pathc - 1], RTLD_NOW | RTLD_GLOBAL);
        py_main_t py_main = library ? (py_main_t)dlsym(library, "Py_BytesMain") : NULL;
        if (py_main != NULL) {
            /* The runtime's own home, and its python3 as sys.executable: a subprocess started with
               sys.executable must be Python, not this app again. */
            setenv("PYTHONHOME", runtime, 1);
            setenv("PYTHONEXECUTABLE", python, 1);
            return py_main(argc + 1, args);
        }
    }
    args[0] = python;
    execv(python, args);
    perror("ChartLibre: could not start Python");
    return 1;
}
